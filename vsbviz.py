import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap

import datasets
import models
import utils

from pathlib import Path


def load_dann_model_and_device(path: str, input_shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        weights = torch.load(path, map_location="cpu")
    else:
        weights = torch.load(path)
    model = models.DomainAdversarialLSTM(input_shape=input_shape)
    model.load_state_dict(weights["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def load_naive_model_and_device(fold: int, input_shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        weights = torch.load(f"output/004_vsb_naive/fold{fold}/checkpoints/best.pth", map_location="cpu")
    else:
        weights = torch.load(f"output/004_vsb_naive/fold{fold}/checkpoints/best.pth")
    model = models.NaiveClassificationLSTM(input_shape=input_shape)
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
    SAVE_DIR = Path("assets/vsb")
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    utils.set_seed(2019)

    # Inference on test and output
    test_dataset = datasets.VSBTestDataset()
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256)
    oofs_dict: dict = {
        "warmup": [],
        "no_warmup": [],
        "naive": []
    }
    oof_labels = []
    predictions_dict: dict = {
        "warmup": [],
        "no_warmup": [],
        "naive": []
    }
    for i in range(5):
        train_dataset = datasets.VSBTrainDataset(fold=i)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=256)

        path = f"output/001_vsb/fold{i}/checkpoints/best.pth"
        input_shape = test_dataset.X_test.shape
        model_warmup, device = load_dann_model_and_device(path, input_shape)

        path = f"output/005_vsb_no_warmup/fold{i}/checkpoints/best.pth"
        model_no_warmup, _ = load_dann_model_and_device(path, input_shape)

        model_naive, _ = load_naive_model_and_device(fold=i, input_shape=input_shape)

        # Out-of-Folds
        for batch, label in train_loader:
            batch = batch.to(device)
            with torch.no_grad():
                output_warmup = model_warmup(batch, 1.0)
                output_no_warmup = model_no_warmup(batch, 1.0)
                output_naive = model_naive(batch)

            oof_labels.append(label.cpu().numpy().reshape(-1))

            prediction_warmup = torch.sigmoid(
                output_warmup["logits"].detach().cpu().numpy().reshape(-1))
            prediction_no_warmup = torch.sigmoid(
                output_no_warmup["logits"].detach().cpu().numpy().reshape(-1))
            prediction_naive = torch.sigmoid(
                output_naive["logits"].detach().cpu().numpy().reshape(-1))

            oofs_dict["warmup"].append(prediction_warmup)
            oofs_dict["no_warmup"].append(prediction_no_warmup)
            oofs_dict["naive"].append(prediction_naive)

        preds: dict = {
            "warmup": [],
            "no_warmup": [],
            "naive": []
        }
        for batch in test_loader:
            batch = batch.to(device)
            with torch.no_grad():
                output_warmup = model_warmup(batch, 1.0)
                output_no_warmup = model_no_warmup(batch, 1.0)
                output_naive = model_naive(batch)

            prediction_warmup = torch.sigmoid(
                output_warmup["logits"].detach().cpu().numpy().reshape(-1))
            prediction_no_warmup = torch.sigmoid(
                output_no_warmup["logits"].detach().cpu().numpy().reshape(-1))
            prediction_naive = torch.sigmoid(
                output_naive["logits"].detach().cpu().numpy().reshape(-1))

            preds["warmup"].append(prediction_warmup)
            preds["no_warmup"].append(prediction_no_warmup)
            preds["naive"].append(prediction_naive)

        for key in preds:
            pred_3 = []
            pred_np = np.concatenate(preds[key])
            for pred_scalar in pred_np:
                for i in range(3):
                    pred_3.append(pred_scalar)
            predictions_dict[key].append(pred_3)

    oof_target = np.concatenate(oof_labels)
    submission = pd.read_csv("input/vsb-power-line-fault-detection/sample_submission.csv")
    output_path = {
        "warmup": Path("output/001_vsb"),
        "no_warmup": Path("output/005_vsb_no_warmup"),
        "naive": Path("output/004_vsb_naive")
    }
    for key in oofs_dict:
        oof_prediction = np.concatenate(oofs_dict[key])
        search_result = utils.threshold_search(oof_target, oof_prediction)
        print(f"{key}", search_result)

        soft_prediction = np.squeeze(np.mean(predictions_dict[key], axis=0))
        hard_prediction = (soft_prediction > search_result["threshold"]).astype(int)
        submission["target"] = hard_prediction
        submission.to_csv(output_path[key] / "submission.csv", index=False)

    representations_dict: dict = {
        "warmup": [],
        "no_warmup": [],
        "naive": []
    }
    labels = []
    domain_labels = []
    for i in range(5):
        dataset = datasets.VSBDataset(mode="valid", fold=i)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256)

        path = f"output/001_vsb/fold{i}/checkpoints/best.pth"
        input_shape = dataset.X.shape
        model_warmup, device = load_dann_model_and_device(path, input_shape)

        path = f"output/005_vsb_no_warmup/fold{i}/checkpoints/best.pth"
        model_no_warmup, _ = load_dann_model_and_device(path, input_shape)

        model_naive, _ = load_naive_model_and_device(fold=i, input_shape=input_shape)

        representations, label, domain_label = get_representation(loader, model_warmup, device)
        representations_dict["warmup"].append(representations)
        labels.append(label)
        domain_labels.append(domain_label)

        representations, _, _ = get_representation(loader, model_no_warmup, device)
        representations_dict["no_warmup"].append(representations)

        representations, _, _ = get_representation(loader, model_naive, device)
        representations_dict["naive"].append(representations)

    labels = np.concatenate(labels)
    domain_labels = np.concatenate(domain_labels)
    domain_labels = np.array(["source" if i == 0 else "target"])
    class_map = {0: "ok", 1: "ng", -1: "target"}
    labels = np.array(list(map(lambda x: class_map[x], labels.tolist())))  # type: ignore

    for key in representations_dict:
        representations = np.concatenate(representations_dict[key])
        umap_plot(representations, domain_labels, save_dir=SAVE_DIR, name=f"umap_domain_{key}.png")
        umap_plot(representations, labels, save_dir=SAVE_DIR, name=f"umap_classes_{key}.png")
