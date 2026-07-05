import torch
import torch.nn as nn
from torchvision import models


class FrameCNNTemporalAvg(nn.Module):
    """ResNet18 features per frame, averaged over time. My simplest baseline."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3, num_classes: int = 2):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
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
        feats = feats.mean(dim=1)
        return self.classifier(feats)

    def param_groups(self, lr: float, backbone_lr: float = None):
        """Same LR everywhere for this simple model, kept for API symmetry
        with the other models that DO need differential learning rates."""
        return [{"params": self.parameters(), "lr": lr}]
