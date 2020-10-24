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
    if isinstance(global_params["seed"], list):
        output_dir = output_dir / "multirun"
        seeds = global_params["seed"]
        multirun = True
    else:
        seeds = [global_params["seed"]]
        multirun = False
    output_dir.mkdir(parents=True, exist_ok=True)

    device = training.get_device(global_params["device"])

    for seed in seeds:
        utils.set_seed(seed)

        dataset_name = config["datasets"]["name"]
        if dataset_name == "digits":
            nfold = 1
        else:
            nfold = 5

        for fold in range(nfold):
            if dataset_name == "digits":
                train_dataset = datasets.DigitsDataset(mode="train")
                valid_dataset = datasets.DigitsDataset(mode="test")
            elif dataset_name == "vsb":
                train_dataset = datasets.VSBDataset(mode="train", fold=fold, seed=seed)  # type: ignore
                valid_dataset = datasets.VSBDataset(mode="valid", fold=fold, seed=seed)  # type: ignore
            else:
                raise NotImplementedError

            train_loader = torch.utils.data.DataLoader(
                train_dataset, **config["loader"]["params"]["train"])
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset, **config["loader"]["params"]["valid"])
            loaders = {"train": train_loader, "valid": valid_loader}

            model_params = config["model"].get("params", {})
            if model_params is None:
                model_params = {}
            if config["model"]["name"] == "cnn":
                model = models.DomainAdversarialCNN(**model_params)
                criterion = model.get_loss_fn()
            elif config["model"]["name"] == "naivecnn":
                model = models.NaiveClassificationCNN(**model_params)  # type: ignore
                criterion = model.get_loss_fn()  # type: ignore
            elif config["model"]["name"] == "rnn":
                model = models.DomainAdversarialLSTM(  # type: ignore
                    input_shape=train_dataset.X.shape,  # type: ignore
                    **model_params)
                criterion = model.get_loss_fn()
            elif config["model"]["name"] == "naivernn":
                model = models.NaiveClassificationLSTM(  # type: ignore
                    input_shape=train_dataset.X.shape,  # type: ignore
                    **model_params)
                criterion = model.get_loss_fn()  # type: ignore
            else:
                raise NotImplementedError

            optimizer = training.get_optimizer(model, config)
            scheduler = training.get_scheduler(optimizer, config)
            callbacks = training.get_callbacks(config)

            if config["runner"] == "dann":
                runner = training.DANNRunner()
            elif config["runner"] == "naive":
                runner = training.NaiveClassificationRunner()
            else:
                raise NotImplementedError

            if multirun:
                logdir = output_dir / f"seed{seed}/fold{fold}"
            else:
                logdir = output_dir / f"fold{fold}"

            if not args.skip:
                runner.train(
                    model=model,
                    loaders=loaders,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    num_epochs=global_params["num_epochs"],
                    verbose=True,
                    logdir=logdir,
                    callbacks=callbacks,
                    main_metric=global_params["main_metric"],
                    minimize_metric=global_params["minimize_metric"])
