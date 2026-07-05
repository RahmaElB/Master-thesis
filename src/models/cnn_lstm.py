import torch
import torch.nn as nn
from torchvision import models


class FrameCNNLSTM(nn.Module):
    """ResNet18 per-frame features fed into an LSTM to actually model temporal order,
    instead of just averaging like the baseline does."""

    def __init__(
        self,
        pretrained: bool = True,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
        num_classes: int = 2,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, t, -1)

        _, (h_n, _) = self.lstm(feats)

        if self.bidirectional:
            final_feat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final_feat = h_n[-1]

        return self.classifier(final_feat)

    def param_groups(self, lr: float, backbone_lr: float = None):
        if backbone_lr is None:
            return [{"params": self.parameters(), "lr": lr}]
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.lstm.parameters(), "lr": lr},
            {"params": self.classifier.parameters(), "lr": lr},
        ]
