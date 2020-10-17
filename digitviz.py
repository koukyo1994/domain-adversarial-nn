import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap

import datasets
import models

from pathlib import Path
from PIL import Image


def plot_mnist_and_mnistm(dataset, save_dir: Path):
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))
    for i in range(10):
        axes[0, i].set_title(f"Class {i}")

    axes[0, 0].set_ylabel("MNIST", rotation=90, size="large")
    axes[1, 0].set_ylabel("MNISTM", rotation=90, size="large")

    master_df: pd.DataFrame = dataset.master_df
    for i in range(10):
        mnist_sample = master_df.query(
            f"source == 'mnist' & label == {i}").sample(1).reset_index(drop=True)
        mnistm_sample = master_df.query(
            f"source == 'mnistm' & label == {i}").sample(1).reset_index(drop=True)

        mnist_img_idx = int(mnist_sample.loc[0, "image"].replace(".png", ""))
        mnist_image = dataset.mnist_images[mnist_img_idx]
        axes[0, i].imshow(mnist_image, cmap="gray")

        mnistm_img_path = dataset.mnistm_images_dir / mnistm_sample.loc[0, "image"]
        image = Image.open(mnistm_img_path).convert("RGB")
        axes[1, i].imshow(image)

        axes[0, i].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        axes[1, i].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(save_dir / "mnist_and_mnistm.png")


def load_dann_model_and_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        weights = torch.load("output/000_digits/checkpoints/best.pth", map_location="cpu")
    else:
        weights = torch.load("output/000_digits/checkpoints/best.pth")
    model = models.DomainAdversarialCNN()
    model.load_state_dict(weights["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def load_naive_model_and_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        weights = torch.load("output/002_digits_naive/checkpoints/best.pth", map_location="cpu")
    else:
        weights = torch.load("output/002_digits_naive/checkpoints/best.pth")
    model = models.NaiveClassificationCNN()
    model.load_state_dict(weights["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def get_representation(loader, model, device):
    model.eval()
    representations = []
    labels = []
    domain_labels = []
    for image, label, domain_label in loader:
        image = image.to(device)
        labels.append(label.cpu().numpy().reshape(-1))
        domain_labels.append(domain_label.cpu().numpy().reshape(-1))
        with torch.no_grad():
            batch_size = image.size(0)
            output = model.feature_extractor(image).view(batch_size, -1)
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
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)

    emb0 = transformer.embedding_[:, 0]
    emb1 = transformer.embedding_[:, 1]
    colors = [
        "red", "green", "blue", "orange", "purple", "brown", "fuchsia", "grey",
        "olive", "lightblue"
    ]

    unique_groups = np.unique(groups)
    for i, group in enumerate(unique_groups):
        group_mask = groups == group
        group_emb0 = emb0[group_mask]
        group_emb1 = emb1[group_mask]
        ax.scatter(group_emb0, group_emb1, c=colors[i], label=str(group), alpha=0.5)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig(save_dir / name)


if __name__ == "__main__":
    SAVE_DIR = Path("assets/digits")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    dataset = datasets.DigitsDataset(mode="test")
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    plot_mnist_and_mnistm(dataset, SAVE_DIR)

    model, device = load_dann_model_and_device()
    representations, labels, domain_labels = get_representation(
        loader, model, device)

    domain_labels = np.array(["mnist" if i == 0 else "mnistm" for i in domain_labels])
    umap_plot(representations, domain_labels, save_dir=SAVE_DIR, name="umap_domain.png")
    umap_plot(representations, labels, save_dir=SAVE_DIR, name="umap_classes.png")

    naive_model, device = load_naive_model_and_device()
    representations, labels, domain_labels = get_representation(
        loader, naive_model, device)
    domain_labels = np.array(["mnist" if i == 0 else "mnistm" for i in domain_labels])
    umap_plot(representations, domain_labels, save_dir=SAVE_DIR, name="umap_domain_naive.png")
    umap_plot(representations, labels, save_dir=SAVE_DIR, name="umap_classes_naive.png")
