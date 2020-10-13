import pandas as pd
import torch
import torch.utils.data as torchdata

from pathlib import Path
from PIL import Image

from torchvision import datasets, transforms


def mono_to_color(x: torch.Tensor):
    return x.repeat(3, 1, 1)


class DigitsDataset(torchdata.Dataset):
    __MODES__ = ["train", "test"]

    def __init__(self, mode="train"):
        assert mode in self.__MODES__, \
            "`mode` should be either 'train' or 'test'"

        self.mode = mode

        if mode == "train":
            mnist = datasets.MNIST(
                "input/digits/mnist", train=True, download=True)
            self.mnistm_images_dir = Path("input/digits/mnist_m/mnist_m_train")
            mnistm_labels = pd.read_csv(
                "input/digits/mnist_m/mnist_m_train_labels.txt",
                sep=" ",
                header=None)
        elif mode == "test":
            mnist = datasets.MNIST(
                "input/digits/mnist", train=False, download=True)
            self.mnistm_images_dir = Path("input/digits/mnist_m/mnist_m_test")
            mnistm_labels = pd.read_csv(
                "input/digits/mnist_m/mnist_m_test_labels.txt",
                sep=" ",
                header=None)
        else:
            raise NotImplementedError

        self.mnist_images = mnist.data.numpy()
        self.mnist_labels = mnist.targets
        mnistm_labels.columns = ["image", "label"]

        self.mnist_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(mono_to_color)
        ])
        self.mnistm_transform = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.ToTensor()
        ])

        master_df = pd.DataFrame(columns=["image", "label", "source"])
        master_df["image"] = [
            f"{i}.png"
            for i in range(len(self.mnist_images))
        ]
        master_df["label"] = self.mnist_labels.numpy()
        master_df["source"] = "mnist"

        mnistm_labels["source"] = "mnistm"
        master_df = pd.concat(
            [master_df, mnistm_labels],
            axis=0,
            sort=False).reset_index(drop=True)
        self.master_df = master_df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, idx: int):
        sample = self.master_df.loc[idx]
        if sample.source == "mnist":
            image_idx = int(sample.image.replace(".png", ""))
            image = self.mnist_images[image_idx]
            image = self.mnist_transform(Image.fromarray(image))
            label = sample.label
            domain_label = 0
        else:
            image_path = self.mnistm_images_dir / sample.image
            image = Image.open(image_path).convert("RGB")
            image = self.mnistm_transform(image)
            label = sample.label
            domain_label = 1
        return image, label, domain_label
