import warnings

import torch

import datasets
import models
import training
import utils

from pathlib import Path


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)

    args = utils.get_parser().parse_args()
    config = utils.load_config(args.config)

    global_params = config["globals"]

    output_dir = Path(global_params["output_dir"])
    config_name = Path(args.config).name.replace(".yml", "")
    output_dir = output_dir / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    utils.set_seed(global_params["seed"])
    device = training.get_device(global_params["device"])

    if config["datasets"]["name"] == "digits":
        train_dataset = datasets.DigitsDataset(mode="train")
        valid_dataset = datasets.DigitsDataset(mode="test")
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset, **config["loader"]["params"]["train"])
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, **config["loader"]["params"]["valid"])
    loaders = {"train": train_loader, "valid": valid_loader}

    if config["model"]["name"] == "cnn":
        model = models.DomainAdversarialCNN()
        criterion = model.get_loss_fn()
    else:
        raise NotImplementedError

    optimizer = training.get_optimizer(model, config)
    scheduler = training.get_scheduler(optimizer, config)
    callbacks = training.get_callbacks(config)

    runner = training.DANNRunner()
    runner.train(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=global_params["num_epochs"],
        verbose=True,
        logdir=output_dir,
        callbacks=callbacks,
        main_metric=global_params["main_metric"],
        minimize_metric=global_params["minimize_metric"])
