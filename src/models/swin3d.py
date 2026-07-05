import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t, Swin3D_T_Weights


class Swin3DClassifier(nn.Module):
    """
    Video Swin Transformer (tiny).

    Note on the training collapse from the first version of this project:
    with a single Adam LR of 1e-4 applied to the whole pretrained network at
    batch size 2, the model never learned anything, val accuracy and val F1
    were IDENTICAL every single epoch (always predicting the majority class),
    and val AUC hovered around 0.55-0.59, i.e. barely better than chance.

    That's the classic signature of a large pretrained model collapsing to
    the trivial majority-class solution: the learning rate was too high for
    full fine-tuning, gradients at batch size 2 are noisy, and there was no
    warmup or gradient clipping to protect the pretrained weights during the
    first few updates. This class exposes separate backbone/head parameter
    groups (see param_groups) and a freeze/unfreeze helper so train.py can:
      1) train only the new classification head for a few epochs
         (backbone frozen, pretrained features untouched), then
      2) unfreeze the backbone and fine-tune everything with a much smaller
         backbone learning rate than the head learning rate.
    Combined with LR warmup + gradient clipping in the training loop, this
    is the standard recipe for fine-tuning large pretrained transformers on
    a small-ish dataset without them collapsing.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3, num_classes: int = 2):
        super().__init__()

        weights = Swin3D_T_Weights.DEFAULT if pretrained else None
        self.model = swin3d_t(weights=weights)

        in_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def backbone_parameters(self):
        return (p for name, p in self.model.named_parameters() if not name.startswith("head."))

    def head_parameters(self):
        return self.model.head.parameters()

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone_parameters():
            p.requires_grad = trainable

    def param_groups(self, lr: float, backbone_lr: float = None):
        """
        lr           -> learning rate for the (newly initialised) head
        backbone_lr  -> much smaller learning rate for the pretrained backbone
        """
        if backbone_lr is None:
            return [{"params": self.parameters(), "lr": lr}]
        return [
            {"params": self.backbone_parameters(), "lr": backbone_lr},
            {"params": self.head_parameters(), "lr": lr},
        ]
