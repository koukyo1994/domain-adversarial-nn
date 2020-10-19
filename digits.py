import warnings

import numpy as np
import torch

import datasets
import models
import training
import utils

from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


def train_one_epoch(loader,
                    model,
                    optimizer,
                    scheduler,
                    epoch: int,
                    device,
                    domain_classification_steps: int,
                    n_steps_for_domain_classifier: int,
                    n_steps_for_feedback: int):
    loss_meter = utils.AverageMeter()
    source_accuracy_meter = utils.AverageMeter()
    target_accuracy_meter = utils.AverageMeter()
    y_preds = []
    y_trues = []
    y_domains = []

    model.train()
    tqdm_bar = tqdm(loader)
    for step, (image, label, domain_label) in enumerate(tqdm_bar):
        image = image.to(device)
        label = label.to(device)
        domain_label = domain_label.to(device)

        output = model(image)
        loss = model.classification_loss(output, label, domain_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=image.size(0))

        y_pred = torch.softmax(output, dim=1).detach().cpu().numpy().argmax(
            axis=1).reshape(-1)
        y_true = label.detach().cpu().numpy().reshape(-1)
        y_domain = domain_label.detach().cpu().numpy().reshape(-1)

        y_source_pred = y_pred[y_domain == 0]
        y_source_target = y_true[y_domain == 0]
        source_acc = accuracy_score(y_true=y_source_target, y_pred=y_source_pred)
        source_accuracy_meter.update(source_acc, n=1)

        y_target_pred = y_pred[y_domain == 1]
        y_target_target = y_true[y_domain == 1]
        target_acc = accuracy_score(y_true=y_target_target, y_pred=y_target_pred)
        target_accuracy_meter.update(target_acc, n=1)

        y_preds.append(y_pred)
        y_trues.append(y_true)
        y_domains.append(y_domain)

        description = f"Step: [{step + 1}/{len(loader)}] " + \
            f"loss: {loss_meter.val:.4f} loss (avg): {loss_meter.avg:.4f} " + \
            f"source acc: {source_accuracy_meter.val:.4f} source acc (avg): {source_accuracy_meter.avg:.4f} " + \
            f"target acc: {target_accuracy_meter.val:.4f} target acc (avg): {target_accuracy_meter.avg:.4f}"
        tqdm_bar.set_description(description)

        if step + 1 % domain_classification_steps == 0:
            phase = {0: "train domain classifier", 1: "feedback for feature extractor"}
            for phase_id in phase:
                print(f"{phase[phase_id]}")
                iterator = iter(loader)
                if phase_id == 0:
                    domain_tqdm_bar = tqdm(range(n_steps_for_domain_classifier))
                    model.prepare_for_domain_classification()
                else:
                    domain_tqdm_bar = tqdm(range(n_steps_for_feedback))
                    model.feedback_for_feature_extractor()

                for dstep in domain_tqdm_bar:
                    image, _, domain_label = next(iterator)
                    image = image.to(device)
                    domain_label = domain_label.to(device)

                    y = model.classify_domain(image, alpha=1.0)
                    loss = model.domain_classification_loss(y, domain_label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    y_pred = torch.sigmoid(y).detach().cpu().numpy()
                    y_true = domain_label.detach().cpu().numpy()
                    score = roc_auc_score(y_true=y_true, y_score=y_pred)
                    domain_tqdm_bar.set_description(
                        f"Step: [{dstep + 1}/{n_steps_for_domain_classifier}] loss: {loss.item():.4f} " +
                        f"AUC: {score:.4f}")

    if scheduler is not None:
        scheduler.step()

    prediction = np.concatenate(y_preds)
    targets = np.concatenate(y_trues)
    domain_targets = np.concatenate(y_domains)

    source_preds = prediction[domain_targets == 0]
    source_targs = targets[domain_targets == 0]
    source_acc = accuracy_score(y_true=source_targs, y_pred=source_preds)

    target_preds = prediction[domain_targets == 1]
    target_targs = targets[domain_targets == 1]
    target_acc = accuracy_score(y_true=target_targs, y_pred=target_preds)

    print(f"Source Accuracy: {source_acc:.4f} Target Accuracy: {target_acc:.4f}")


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

    train_dataset = datasets.DigitsDataset(mode="train")
    valid_dataset = datasets.DigitsDataset(mode="test")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, **config["loader"]["params"]["train"])
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, **config["loader"]["params"]["valid"])

    model = models.TrainInTurnsDANNCNN()

    optimizer = training.get_optimizer(model, config)
    scheduler = training.get_scheduler(optimizer, config)

    for epoch in range(global_params["num_epochs"]):
        print(f"Epoch: [{epoch + 1}/{global_params['num_epochs']}]")
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch=epoch + 1,
            device=device,
            domain_classification_steps=config["training"]["domain_classification_steps"],
            n_steps_for_domain_classifier=config["training"]["n_steps_for_domain_classifier"],
            n_steps_for_feedback=config["training"]["n_steps_for_feedback"])
