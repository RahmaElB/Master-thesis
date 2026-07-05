import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class R3DClassifier(nn.Module):
    """3D ResNet-18, processes the clip as a spatio-temporal volume directly."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3, num_classes: int = 2):
        super().__init__()

        weights = R3D_18_Weights.DEFAULT if pretrained else None
        self.model = r3d_18(weights=weights)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def backbone_parameters(self):
        return (p for name, p in self.model.named_parameters() if not name.startswith("fc."))

    def head_parameters(self):
        return self.model.fc.parameters()

    def param_groups(self, lr: float, backbone_lr: float = None):
        if backbone_lr is None:
            return [{"params": self.parameters(), "lr": lr}]
        return [
            {"params": self.backbone_parameters(), "lr": backbone_lr},
            {"params": self.head_parameters(), "lr": lr},
        ]

    def last_conv_layer(self):
        """Used by Grad-CAM: last 3D conv block before global pooling."""
        return self.model.layer4[-1].conv2[0]
