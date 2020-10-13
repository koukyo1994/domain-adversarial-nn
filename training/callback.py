import numpy as np
import torch

from typing import List

from catalyst.core import Callback, CallbackOrder, State
from sklearn.metrics import accuracy_score, roc_auc_score


def get_callbacks(config: dict):
    callbacks = []  # type: ignore
    required_callbacks = config["callbacks"]
    if required_callbacks is None:
        return callbacks
    else:
        for callback_conf in required_callbacks:
            name = callback_conf["name"]
            params = callback_conf["params"]
            callback_cls = globals().get(name)

            if callback_cls is not None:
                callbacks.append(callback_cls(**params))
        return callbacks


class DANNClassificationAccuracy(Callback):
    def __init__(self, output_key="logits",
                 prefixes={0: "source_acc", 1: "target_acc"}):
        super().__init__(order=CallbackOrder.Metric)

        self.output_key = output_key
        self.prefixes = prefixes

    def on_loader_start(self, state: State):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []
        self.domain_target: List[np.ndarray] = []

    def on_batch_end(self, state: State):
        targ = state.input[1].detach().cpu().numpy()
        domain_targ = state.input[2].detach().cpu().numpy()

        out = state.output[self.output_key]

        classification_result = torch.softmax(
            out["logits"].detach(), dim=1).cpu().numpy()

        self.prediction.append(classification_result)
        self.target.append(targ)
        self.domain_target.append(domain_targ)

        domain_0_pred = classification_result[domain_targ == 0].argmax(axis=1)
        domain_1_pred = classification_result[domain_targ == 1].argmax(axis=1)

        domain_0_targ = targ[domain_targ == 0]
        domain_1_targ = targ[domain_targ == 1]

        domain_0_score = accuracy_score(
            y_true=domain_0_targ, y_pred=domain_0_pred)
        domain_1_score = accuracy_score(
            y_true=domain_1_targ, y_pred=domain_1_pred)

        state.batch_metrics[self.prefixes[0]] = domain_0_score
        state.batch_metrics[self.prefixes[1]] = domain_1_score

    def on_loader_end(self, state: State):
        prediction = np.concatenate(self.prediction, axis=0)
        target = np.concatenate(self.target, axis=0)
        domain_target = np.concatenate(self.domain_target, axis=0)

        domain_0_pred = prediction[domain_target == 0].argmax(axis=1)
        domain_1_pred = prediction[domain_target == 1].argmax(axis=1)

        domain_0_targ = target[domain_target == 0]
        domain_1_targ = target[domain_target == 1]

        domain_0_score = accuracy_score(
            y_true=domain_0_targ, y_pred=domain_0_pred)
        domain_1_score = accuracy_score(
            y_true=domain_1_targ, y_pred=domain_1_pred)

        state.loader_metrics[self.prefixes[0]] = domain_0_score
        state.loader_metrics[self.prefixes[1]] = domain_1_score

        if state.is_valid_loader:
            state.epoch_metrics[state.valid_loader + "_epoch_" +
                                self.prefixes[0]] = domain_0_score
            state.epoch_metrics[state.valid_loader + "_epoch_" +
                                self.prefixes[1]] = domain_1_score
        else:
            state.epoch_metrics[
                "train_epoch_" + self.prefixes[0]] = domain_0_score
            state.epoch_metrics[
                "train_epoch_" + self.prefixes[1]] = domain_1_score


class DANNDomainAUC(Callback):
    def __init__(self, output_key="logits",
                 prefix="domain_auc"):
        super().__init__(order=CallbackOrder.Metric)

        self.output_key = output_key
        self.prefix = prefix

    def on_loader_start(self, state: State):
        self.prediction: List[np.ndarray] = []
        self.domain_target: List[np.ndarray] = []

    def on_batch_end(self, state: State):
        domain_targ = state.input[2].detach().cpu().numpy()
        out = state.output[self.output_key]

        domain_classification = torch.sigmoid(
            out["domain_logits"].detach()).cpu().numpy()

        self.prediction.append(domain_classification)
        self.domain_target.append(domain_targ)

        score = roc_auc_score(
            y_true=domain_targ.reshape(-1),
            y_score=domain_classification.reshape(-1))
        state.batch_metrics[self.prefix] = score

    def on_loader_end(self, state: State):
        y_pred = np.concatenate(self.prediction, axis=0).reshape(-1)
        y_true = np.concatenate(self.domain_target, axis=0).reshape(-1)

        score = roc_auc_score(
            y_true=y_true,
            y_score=y_pred)
        state.loader_metrics[self.prefix] = score
        if state.is_valid_loader:
            state.epoch_metrics[state.valid_loader + "_epoch_" +
                                self.prefix] = score
        else:
            state.epoch_metrics["train_epoch_" + self.prefix] = score
