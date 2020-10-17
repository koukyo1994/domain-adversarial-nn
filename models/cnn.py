import torch
import torch.nn as nn

from .layer import GradientReversalLayer


class DomainAdversarialCNN(nn.Module):
    def __init__(self, img_size=32):
        super().__init__()
        self.img_size = img_size
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.Conv2d(50, 50, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU()
        )

        in_features = self._get_in_features()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def _get_in_features(self):
        in_channels = self.feature_extractor[0].in_channels
        dummy = torch.ones((1, in_channels, self.img_size, self.img_size))
        out = self.feature_extractor(dummy)
        return out.size(1) * (out.size(2) ** 2)

    def forward(self, x, alpha):
        batch_size = x.size(0)
        x = self.feature_extractor(x).view(batch_size, -1)
        y = GradientReversalLayer.apply(x, alpha)
        x = self.classifier(x)
        y = self.domain_classifier(y)
        return {
            "logits": x.view(batch_size, -1),
            "domain_logits": y.view(batch_size, -1)
        }

    def get_loss_fn(self):
        class DANNLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.loss_classifier = nn.CrossEntropyLoss()
                self.loss_domain = nn.BCEWithLogitsLoss()

            def forward(self, x, y, target, domain_target):
                source_preds = x[domain_target == 0]
                source_target = target[domain_target == 0]

                source_classification_loss = self.loss_classifier(
                    source_preds, source_target)

                domain_classification_loss = self.loss_domain(
                    y.view(-1), domain_target.float())
                return source_classification_loss + domain_classification_loss
        return DANNLoss()


class NaiveClassificationCNN(nn.Module):
    def __init__(self, img_size=32):
        super().__init__()
        self.img_size = img_size
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.Conv2d(50, 50, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU()
        )

        in_features = self._get_in_features()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def _get_in_features(self):
        in_channels = self.feature_extractor[0].in_channels
        dummy = torch.ones((1, in_channels, self.img_size, self.img_size))
        out = self.feature_extractor(dummy)
        return out.size(1) * (out.size(2) ** 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extractor(x).view(batch_size, -1)
        x = self.classifier(x)
        return {
            "logits": x.view(batch_size, -1)
        }

    def get_loss_fn(self):
        class NaiveClassificationLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.loss_classifier = nn.CrossEntropyLoss()

            def forward(self, x, target, domain_target):
                source_preds = x[domain_target == 0]
                source_target = target[domain_target == 0]

                source_classification_loss = self.loss_classifier(
                    source_preds, source_target)

                return source_classification_loss
        return NaiveClassificationLoss()
