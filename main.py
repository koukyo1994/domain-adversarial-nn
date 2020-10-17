import warnings

import numpy as np
import pandas as pd
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
            train_dataset = datasets.VSBDataset(mode="train", fold=fold)  # type: ignore
            valid_dataset = datasets.VSBDataset(mode="valid", fold=fold)  # type: ignore
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
        elif config["model"]["name"] == "rnn":
            model = models.DomainAdversarialLSTM(input_shape=train_dataset.X.shape)  # type: ignore
            criterion = model.get_loss_fn()
        else:
            raise NotImplementedError

        optimizer = training.get_optimizer(model, config)
        scheduler = training.get_scheduler(optimizer, config)
        callbacks = training.get_callbacks(config)

        runner = training.DANNRunner()
        if not args.skip:
            runner.train(
                model=model,
                loaders=loaders,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=global_params["num_epochs"],
                verbose=True,
                logdir=output_dir / f"fold{fold}",
                callbacks=callbacks,
                main_metric=global_params["main_metric"],
                minimize_metric=global_params["minimize_metric"])

    if dataset_name == "vsb":
        oofs = []
        oof_labels = []
        predictions = []
        test_dataset = datasets.VSBTestDataset()
        test_loader = torch.utils.data.DataLoader(
            test_dataset, **config["loader"]["params"]["valid"])
        for fold in range(5):
            oof_dataset = datasets.VSBTrainDataset(fold=fold)
            oof_loader = torch.utils.data.DataLoader(
                oof_dataset, **config["loader"]["params"]["valid"])

            weights = torch.load(output_dir / f"fold{fold}/checkpoints/best.pth")
            model.load_state_dict(weights["model_state_dict"])
            model.to(device)
            model.eval()
            preds = []
            for batch, label in oof_loader:
                batch = batch.to(device)
                with torch.no_grad():
                    output = model(batch, 1.0)
                prediction = torch.sigmoid(output["logits"].detach()).cpu().numpy().reshape(-1)
                oofs.append(prediction)
                oof_labels.append(label.cpu().numpy().reshape(-1))
            for batch in test_loader:
                batch = batch.to(device)
                with torch.no_grad():
                    output = model(batch, 1.0)
                prediction = torch.sigmoid(output["logits"].detach()).cpu().numpy().reshape(-1)
                preds.append(prediction)
            preds = np.concatenate(preds)
            pred_3 = []
            for pred_scalar in preds:
                for i in range(3):
                    pred_3.append(pred_scalar)
            predictions.append(pred_3)

        oof_prediction = np.concatenate(oofs)
        oof_target = np.concatenate(oof_labels)

        search_result = utils.threshold_search(oof_target, oof_prediction)
        print(search_result)

        soft_prediction = np.squeeze(np.mean(predictions, axis=0))
        hard_prediction = (soft_prediction > search_result["threshold"]).astype(int)
        submission = pd.read_csv("input/vsb-power-line-fault-detection/sample_submission.csv")
        submission["target"] = hard_prediction
        submission.to_csv(output_dir / "submission.csv", index=False)
