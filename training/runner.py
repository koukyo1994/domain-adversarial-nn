import numpy as np

from catalyst import dl


class DANNRunner(dl.Runner):
    def predict_batch(self, batch, **kwargs):
        return super().predict_batch(batch, **kwargs)

    def _handle_batch(self, batch):
        if self.is_train_loader:
            total_steps = self.num_epochs * self.loader_len
            p = float(self.loader_batch_step +
                      (self.epoch - 1) * self.loader_len) / total_steps
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        else:
            alpha = 0.5

        X, label, domain_label = batch
        X = X.to(self.device)
        label = label.to(self.device)
        domain_label = domain_label.to(self.device)

        preds = self.model(X, alpha)
        loss = self.criterion(
            preds["logits"],
            preds["domain_logits"],
            label,
            domain_label)

        self.batch_metrics.update({"loss": loss})
        self.output = {
            "logits": preds
        }

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
