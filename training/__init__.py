import torch
import torch.nn as nn
import torch.optim as optim

from .callback import get_callbacks
from .runner import DANNRunner


def get_device(device: str):
    if torch.cuda.is_available() and "cuda" in device:
        return torch.device(device)
    else:
        return torch.device("cpu")


def get_optimizer(model: nn.Module, config: dict):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name")

    return optim.__getattribute__(optimizer_name)(model.parameters(),
                                                  **optimizer_config["params"])


def get_scheduler(optimizer, config: dict):
    scheduler_config = config["scheduler"]
    scheduler_name = scheduler_config.get("name")

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **scheduler_config["params"])
