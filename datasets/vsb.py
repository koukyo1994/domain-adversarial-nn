import numpy as np
import pandas as pd
import torch.utils.data as torchdata

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


_MAX_NUM = 127
_MIN_NUM = -128
_SAMPLE_SIZE = 800000


def min_max_transf(ts: np.ndarray,
                   min_data: int,
                   max_data: int,
                   range_needed=(-1, 1)):
    if min_data < 0:
        ts_std = (ts + abs(min_data) / (max_data + abs(min_data)))
    else:
        ts_std = (ts - min_data) / (max_data - min_data)
    if range_needed[0] < 0:
        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


def transform_ts(ts: np.ndarray, n_dim=160, min_max=(-1, 1)):
    ts_std = min_max_transf(ts, min_data=_MIN_NUM, max_data=_MAX_NUM)
    bucket_size = int(_SAMPLE_SIZE / n_dim)
    new_ts = []
    for i in range(0, _SAMPLE_SIZE, bucket_size):
        ts_range = ts_std[i: i + bucket_size]
        mean = ts_range.mean()
        std = ts_range.std()
        std_top = mean + std
        std_bot = mean - std
        percentile_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100])
        max_range = percentile_calc[-1] - percentile_calc[0]
        relative_percentile = percentile_calc - mean
        new_ts.append(
            np.concatenate([
                np.asarray([mean, std, std_top, std_bot, max_range]),
                percentile_calc,
                relative_percentile
            ]))
    return np.asarray(new_ts)


def prep_data(metadata: pd.DataFrame, start: int, end: int):
    parq_train = pd.read_parquet(
        "input/vsb-power-line-fault-detection/train.parquet",
        columns=[str(i) for i in range(start, end)])
    X = []
    y = []
    for id_measurement in tqdm(
            metadata.index.levels[0].unique()[int(start / 3):int(end / 3)]):
        X_signal = []
        for phase in [0, 1, 2]:
            signal_id, target = metadata.loc[id_measurement].loc[phase]
            if phase == 0:
                y.append(target)
            X_signal.append(transform_ts(parq_train[str(signal_id)]))
        X_signal = np.concatenate(X_signal, axis=1)
        X.append(X_signal)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


def calc_train_features():
    metadata = pd.read_csv("input/vsb-power-line-fault-detection/metadata_train.csv")
    metadata = metadata.set_index(["id_measurement", "phase"])
    total_size = len(metadata)
    X = []
    y = []
    for start, end in [(0, int(total_size / 2)), (int(total_size / 2), total_size)]:
        X_temp, y_temp = prep_data(metadata, start, end)
        X.append(X_temp)
        y.append(y_temp)
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y


def calc_test_features():
    metadata = pd.read_csv("input/vsb-power-line-fault-detection/metadata_test.csv")
    metadata = metadata.set_index(["signal_id"])
    first_sig = metadata.index[0]
    n_parts = 10
    max_line = len(metadata)
    part_size = int(max_line / n_parts)
    last_part = max_line % n_parts

    start_end = [
        (x, x + part_size)
        for x in range(first_sig, max_line + first_sig, part_size)
    ]
    start_end = start_end[:-1] + [(start_end[-1][0], start_end[-1][0] + last_part)]
    X_test = []
    for start, end in start_end:
        subset_test = pd.read_parquet(
            "input/vsb-power-line-fault-detection/test.parquet",
            columns=[str(i) for i in range(start, end)])
        for i in tqdm(subset_test.columns):
            id_measurement, phase = metadata.loc[int(i)]
            subset_test_col = subset_test[i]
            subset_trans = transform_ts(subset_test_col)
            X_test.append([i, id_measurement, phase, subset_trans])
    X_test_input = np.asarray([
        np.concatenate([
            X_test[i][3],
            X_test[i + 1][3],
            X_test[i + 2][3]
        ], axis=1)
        for i in range(0, len(X_test), 3)
    ])
    return X_test_input


def save_features(name: str, feats: np.ndarray):
    feature_dir = Path("feature")
    feature_dir.mkdir(exist_ok=True, parents=True)

    np.save(feature_dir / name, feats)


def load_or_calculate_train():
    feature_dir = Path("feature")
    train_feature = feature_dir / "X.npy"
    train_targets = feature_dir / "y.npy"

    if not train_feature.exists() or not train_targets.exists():
        print("Calculate train features")
        X, y = calc_train_features()
        save_features("X.npy", X)
        save_features("y.npy", y)
    else:
        print("Load train features")
        X = np.load(train_feature)
        y = np.load(train_targets)
    return X, y


def load_or_calculate_test():
    feature_dir = Path("feature")
    test_feature = feature_dir / "X_test.npy"

    if not test_feature.exists():
        print("Calculate test features")
        X_test = calc_test_features()
        save_features("X_test.npy", X_test)
    else:
        print("Load test features")
        X_test = np.load(test_feature)
    return X_test


class VSBTestDataset(torchdata.Dataset):
    def __init__(self):
        self.X_test = load_or_calculate_test()

    def __len__(self):
        return len(self.X_test)

    def __getitem__(self, index: int):
        return self.X_test[index].astype(np.float32)


class VSBDataset(torchdata.Dataset):
    __MODES__ = ["train", "valid"]

    def __init__(self, mode="train", fold=0):
        assert mode in self.__MODES__, \
            "`mode` should be either 'train' or 'valid'"

        self.mode = mode
        X, y = load_or_calculate_train()
        self.X_test = load_or_calculate_test()

        trn_idx, val_idx = list(
            StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=2019).split(X, y)
        )[fold]
        if mode == "train":
            self.X = X[trn_idx]
            self.y = y[trn_idx]
        else:
            self.X = X[val_idx]
            self.y = y[val_idx]

        master_df = pd.DataFrame(columns=["idx", "source"])
        master_df["idx"] = list(range(len(self.X)))
        master_df["source"] = "train"

        test_master_df = pd.DataFrame(columns=["idx", "source"])
        test_master_df["idx"] = list(range(len(self.X_test)))
        test_master_df["source"] = "test"
        test_master_df = test_master_df.sample(
            n=len(master_df),
            replace=False,
            random_state=2019 + fold).reset_index(drop=True)

        master_df = pd.concat(
            [master_df, test_master_df],
            axis=0,
            sort=False).reset_index(drop=True)
        self.master_df = master_df.sample(
            frac=1, random_state=2019 + fold).reset_index(drop=True)

    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, idx: int):
        sample = self.master_df.loc[idx]
        domain_idx = sample["idx"]
        if sample.source == "train":
            x = self.X[domain_idx].astype(np.float32)
            y = self.y[domain_idx]
            domain_label = 0
        else:
            x = self.X_test[domain_idx].astype(np.float32)
            y = -1
            domain_label = 1
        return x, y, domain_label
