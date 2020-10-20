import torch.nn as nn

from .layer import Attention, GradientReversalLayer


class DomainAdversarialLSTM(nn.Module):
    def __init__(self, input_shape, hidden_size=128, linear_size=64, n_attention=50, warmup=True):
        super().__init__()

        self.maxlen = input_shape[1]
        self.input_dim = input_shape[2]
        self.warmup = warmup

        self.lstm1 = nn.LSTM(
            self.input_dim, hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(
            hidden_size * 2,
            int(hidden_size / 2),
            bidirectional=True,
            batch_first=True)
        self.attn = Attention(
            int(hidden_size / 2) * 2, self.maxlen, n_attention, n_attention)
        self.lin1 = nn.Linear(int(hidden_size / 2) * 2, linear_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(linear_size, 1)

        self.domain_classifier = nn.Sequential(
            nn.Linear(linear_size, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 1))

    def forward(self, x, alpha):
        batch_size = x.size(0)

        h_lstm, _ = self.lstm1(x)
        h_lstm, _ = self.lstm2(h_lstm)
        attn = self.attn(h_lstm)
        x = self.relu(self.lin1(attn))
        if self.warmup:
            y = GradientReversalLayer.apply(x, alpha)
        else:
            y = GradientReversalLayer.apply(x, 1.0)
        x = self.lin2(x)
        y = self.domain_classifier(y)
        return {
            "logits": x.view(batch_size, -1),
            "domain_logits": y.view(batch_size, -1)
        }

    def get_loss_fn(self):
        class DANNLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.loss = nn.BCEWithLogitsLoss()

            def forward(self, x, y, target, domain_target):
                source_preds = x[domain_target == 0]
                source_target = target[domain_target == 0]

                source_classification_loss = self.loss(
                    source_preds.view(-1), source_target.float())

                domain_classification_loss = self.loss(
                    y.view(-1), domain_target.float())
                return source_classification_loss + domain_classification_loss
        return DANNLoss()


class NaiveClassificationLSTM(nn.Module):
    def __init__(self, input_shape, hidden_size=128, linear_size=64, n_attention=50):
        super().__init__()

        self.maxlen = input_shape[1]
        self.input_dim = input_shape[2]

        self.lstm1 = nn.LSTM(
            self.input_dim, hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(
            hidden_size * 2,
            int(hidden_size / 2),
            bidirectional=True,
            batch_first=True)
        self.attn = Attention(
            int(hidden_size / 2) * 2, self.maxlen, n_attention, n_attention)
        self.lin1 = nn.Linear(int(hidden_size / 2) * 2, linear_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(linear_size, 1)

    def forward(self, x, alpha):
        batch_size = x.size(0)

        h_lstm, _ = self.lstm1(x)
        h_lstm, _ = self.lstm2(h_lstm)
        attn = self.attn(h_lstm)
        x = self.relu(self.lin1(attn))
        x = self.lin2(x)
        return {
            "logits": x.view(batch_size, -1)
        }

    def get_loss_fn(self):
        class NaiveClassificationLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.loss = nn.BCEWithLogitsLoss()

            def forward(self, x, y, target, domain_target):
                source_preds = x[domain_target == 0]
                source_target = target[domain_target == 0]

                source_classification_loss = self.loss(
                    source_preds.view(-1), source_target.float())
                return source_classification_loss
        return NaiveClassificationLoss()
