import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap

import datasets
import models
import utils

from pathlib import Path


def load_model_and_device(path: Path, config: dict, input_shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        weights = torch.load(path, map_location="cpu")
    else:
        weights = torch.load(path)
    model_params = config["model"].get("params")
    if model_params is None:
        model_params = {}

    if config["model"]["name"] == "rnn":
        model = models.DomainAdversarialLSTM(input_shape, **model_params)
    elif config["model"]["name"] == "naivernn":
        model = models.NaiveClassificationLSTM(input_shape, **model_params)  # type: ignore
    else:
        raise NotImplementedError
    model.load_state_dict(weights["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def get_representation(loader, model, device):
    model.eval()
    representations = []
    labels = []
    domain_labels = []
    for data, label, domain_label in loader:
        data = data.to(device)
        labels.append(label.cpu().numpy().reshape(-1))
        domain_labels.append(domain_label.cpu().numpy().reshape(-1))
        with torch.no_grad():
            batch_size = data.size(0)
            output = model.feature_extractor(data).view(batch_size, -1)
        representations.append(output.detach().cpu().numpy())

    representations = np.concatenate(representations, axis=0)
    labels = np.concatenate(labels, axis=0)
    domain_labels = np.concatenate(domain_labels, axis=0)
    return representations, labels, domain_labels


def umap_plot(representations: np.ndarray, groups: np.ndarray, save_dir: Path, name: str):
    transformer = umap.UMAP(
        n_components=2,
        n_neighbors=5,
        random_state=42).fit(representations)
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111)
    ax.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)

    emb0 = transformer.embedding_[:, 0]
    emb1 = transformer.embedding_[:, 1]
    colors = [
        "green", "red", "blue", "orange", "purple", "brown", "fuchsia", "grey",
        "olive", "lightblue"
    ]

    unique_groups = np.unique(groups)
    for i, group in enumerate(unique_groups):
        group_mask = groups == group
        group_emb0 = emb0[group_mask]
        group_emb1 = emb1[group_mask]
        ax.scatter(group_emb0, group_emb1, c=colors[i], label=str(group), alpha=0.5)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(save_dir / name)


if __name__ == "__main__":
    args = utils.get_parser().parse_args()
    config_name = Path(args.config).name.replace(".yml", "")
    config = utils.load_config(args.config)

    global_params = config["globals"]
    if isinstance(global_params["seed"], list):
        multirun = True
        seeds = global_params["seed"]
    else:
        multirun = False
        seeds = [global_params["seed"]]

    EXPERIMENT_BASE_DIR = Path(f"output/{config_name}")
    if multirun:
        EXPERIMENT_BASE_DIR = EXPERIMENT_BASE_DIR / "multirun"

    test_dataset = datasets.VSBTestDataset()
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256)
    for seed in seeds:
        if multirun:
            EXPERIMENT_DIR = EXPERIMENT_BASE_DIR / f"seed{seed}"
        else:
            EXPERIMENT_DIR = EXPERIMENT_BASE_DIR

        oof_labels = []
        oof_preds = []
        fold_test_predictions = []
        for i in range(5):
            train_dataset = datasets.VSBTrainDataset(fold=i, seed=seed)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=256)

            weights_path = EXPERIMENT_DIR / f"fold{i}/checkpoints/best.pth"
            input_shape = test_dataset.X_test.shape

            model, device = load_model_and_device(
                path=weights_path, config=config, input_shape=input_shape)

            # Out-of-folds
            for batch, label in train_loader:
                batch = batch.to(device)
                with torch.no_grad():
                    if isinstance(model, models.DomainAdversarialLSTM):
                        output = model(batch, 1.0)
                    else:
                        output = model(batch)

                oof_labels.append(label.cpu().numpy().reshape(-1))
                prediction = torch.sigmoid(
                    output["logits"]).detach().cpu().numpy().reshape(-1)
                oof_preds.append(prediction)

            # Test data
            test_preds = []
            for batch in test_loader:
                batch = batch.to(device)
                with torch.no_grad():
                    if isinstance(model, models.DomainAdversarialLSTM):
                        output = model(batch, 1.0)
                    else:
                        output = model(batch)

                prediction = torch.sigmoid(
                    output["logits"]).detach().cpu().numpy().reshape(-1)
                test_preds.append(prediction)

            test_preds_np = np.repeat(np.concatenate(test_preds), 3)
            fold_test_predictions.append(test_preds_np)

        oof_target = np.concatenate(oof_labels)
        oof_prediction = np.concatenate(oof_preds)

        search_result = utils.threshold_search(y_true=oof_target, y_proba=oof_prediction)
        print(f"SEED: {seed}", search_result)

        soft_prediction = np.squeeze(np.mean(fold_test_predictions, axis=0))
        hard_prediction = (soft_prediction > search_result["threshold"]).astype(int)

        submission = pd.read_csv("input/vsb-power-line-fault-detection/sample_submission.csv")
        submission["target"] = hard_prediction
        submission.to_csv(EXPERIMENT_DIR / "submission.csv", index=False)

        representations = []
        labels = []
        domain_labels = []
        for i in range(5):
            dataset = datasets.VSBDataset(mode="valid", fold=i, seed=seed)
            loader = torch.utils.data.DataLoader(dataset, batch_size=256)

            weights_path = EXPERIMENT_DIR / f"fold{i}/checkpoints/best.pth"
            input_shape = dataset.X.shape
            model, device = load_model_and_device(
                path=weights_path, config=config, input_shape=input_shape)

            representation, label, domain_label = get_representation(loader, model, device)
            representations.append(representation)
            labels.append(label)
            domain_labels.append(domain_label)

        labels = np.concatenate(labels)
        domain_labels = np.concatenate(domain_labels)
        domain_labels = np.array(["source" if i == 0 else "target" for i in domain_labels])
        class_map = {0: "ok", 1: "ng", -1: "target"}
        labels = np.array(list(map(lambda x: class_map[x], labels.tolist())))  # type: ignore

        representations = np.concatenate(representations, axis=0)
        umap_plot(representations, domain_labels, save_dir=EXPERIMENT_DIR, name="umap_domain.png")
        umap_plot(representations, labels, save_dir=EXPERIMENT_DIR, name="umap_classes.png")
