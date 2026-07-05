"""
Interpretability, requested by my supervisor: "try to work on the
interpretability of the model".

Two methods, because the four models aren't all convolutional:

- Grad-CAM (Selvaraju et al., 2017): works for the baseline, CNN+LSTM and R3D
  models since they all have a normal conv backbone I can hook into. Produces
  a heatmap over each sampled frame showing which regions drove the
  prediction - ideally the left ventricle.

- Occlusion sensitivity: a model-agnostic fallback that works for Swin3D too
  (Grad-CAM needs a conv layer to hook, and hooking into window attention
  blocks is a lot more involved). It works by masking out patches of a frame
  one at a time and measuring how much the predicted probability drops,
  bigger drop = that patch mattered more. Slower, but architecture-agnostic
  and a fair comparison across all four models.
"""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM3D:
    """Grad-CAM for models whose forward pass ends in global pooling over a
    (C, T, H, W) or (B*T, C, H, W)-style conv feature map. Works for R3D
    (3D conv target layer) and the 2D backbones used by baseline/cnn_lstm
    (2D conv target layer, applied per frame).
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        x: a single sample, already batched to shape (1, ...) matching what
           the model's forward() expects (2D-family: (1,T,C,H,W); 3D-family:
           (1,C,T,H,W)).
        Returns a CAM normalized to [0, 1] with shape:
          - (T, H, W) for 3D conv targets
          - (H, W)    for 2D conv targets (per-frame CAMs are not separable
                       from a temporally-averaged 2D backbone, so this method
                       is called once per frame for those models - see
                       compute_gradcam_2d below)
        """
        self.model.zero_grad()
        logits = self.model(x)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())
        score = logits[0, target_class]
        score.backward()

        activations = self.activations[0]  # (C, T, H, W) or (C, H, W)
        gradients = self.gradients[0]

        if activations.dim() == 4:  # (C, T, H, W) - 3D conv
            weights = gradients.mean(dim=(1, 2, 3), keepdim=True)  # (C,1,1,1)
            cam = (weights * activations).sum(dim=0)  # (T, H, W)
        else:  # (C, H, W) - 2D conv
            weights = gradients.mean(dim=(1, 2), keepdim=True)
            cam = (weights * activations).sum(dim=0)  # (H, W)

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), target_class


def compute_gradcam_r3d(model, video_3d: torch.Tensor, target_class: int = None):
    """video_3d: (1, C, T, H, W), the exact input the R3D/Swin3D dataset produces."""
    target_layer = model.last_conv_layer()
    cam_engine = GradCAM3D(model, target_layer)
    cam, pred_class = cam_engine(video_3d, target_class)  # (T, H, W)

    T, H, W = video_3d.shape[2], video_3d.shape[3], video_3d.shape[4]
    cam_resized = np.stack([
        np.array(
            torch.nn.functional.interpolate(
                torch.tensor(cam[t])[None, None], size=(H, W), mode="bilinear", align_corners=False
            )[0, 0]
        )
        for t in range(cam.shape[0])
    ])
    return cam_resized, pred_class  # (T, H, W) in [0,1]


def compute_gradcam_2d_backbone(model, video_2d: torch.Tensor, target_class: int = None):
    """
    For baseline / cnn_lstm: these process frames through a shared 2D ResNet
    backbone and then combine over time (mean or LSTM), so we can't get a
    single clean backward pass CAM per frame with one call - instead we run
    the full model once to get the predicted class, then compute a per-frame
    CAM by backpropagating from the frame's own contribution to that logit.

    video_2d: (1, T, C, H, W)
    Returns per-frame CAMs of shape (T, H, W).
    """
    model.zero_grad()
    logits = model(video_2d)
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())

    target_layer = model.backbone.layer4[-1].conv2
    activations_list: List[torch.Tensor] = []
    gradients_list: List[torch.Tensor] = []

    def fwd_hook(module, inp, out):
        activations_list.append(out)

    def bwd_hook(module, grad_in, grad_out):
        gradients_list.append(grad_out[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    logits = model(video_2d)
    score = logits[0, target_class]
    score.backward()

    h1.remove()
    h2.remove()

    # activations_list[0] has shape (T, C, h, w) since backbone was called on
    # the (B*T, C, H, W) flattened batch and B=1 here.
    activations = activations_list[0]  # (T, C, h, w)
    gradients = gradients_list[0]      # (T, C, h, w)

    weights = gradients.mean(dim=(2, 3), keepdim=True)  # (T, C, 1, 1)
    cams = F.relu((weights * activations).sum(dim=1))   # (T, h, w)

    H, W = video_2d.shape[3], video_2d.shape[4]
    cams_resized = F.interpolate(cams.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)
    cams_resized = cams_resized.squeeze(1).detach().cpu().numpy()

    cams_resized = cams_resized - cams_resized.min()
    cams_resized = cams_resized / (cams_resized.max() + 1e-8)
    return cams_resized, target_class


def occlusion_sensitivity(
    model,
    video: torch.Tensor,
    target_class: int = None,
    patch_size: int = 16,
    stride: int = 16,
    baseline_value: float = 0.0,
) -> np.ndarray:
    """
    Model-agnostic interpretability, used for Swin3D where Grad-CAM isn't a
    good fit. Slides an occluding patch over each sampled frame and records
    how much the predicted probability for target_class drops. Works on
    whatever tensor layout the model expects (2D or 3D family) as long as
    `video` is already the exact model input, shape (1, ...).

    Returns an importance map, one per sampled frame: shape (T, H_frame, W_frame)
    scaled to [0, 1] (higher = more important).
    """
    model.eval()
    with torch.no_grad():
        base_logits = model(video)
        base_probs = F.softmax(base_logits, dim=1)
        if target_class is None:
            target_class = int(torch.argmax(base_probs, dim=1).item())
        base_prob = base_probs[0, target_class].item()

    # figure out layout: (1,T,C,H,W) vs (1,C,T,H,W)
    if video.shape[1] <= 4:  # channel-first 3D layout (1,C,T,H,W)
        _, C, T, H, W = video.shape
        layout = "CTHW"
    else:  # (1,T,C,H,W)
        _, T, C, H, W = video.shape
        layout = "TCHW"

    heatmap = np.zeros((T, H, W), dtype=np.float32)
    counts = np.zeros((T, H, W), dtype=np.float32)

    with torch.no_grad():
        for t in range(T):
            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    y2, x2 = min(y + patch_size, H), min(x + patch_size, W)
                    occluded = video.clone()
                    if layout == "CTHW":
                        occluded[0, :, t, y:y2, x:x2] = baseline_value
                    else:
                        occluded[0, t, :, y:y2, x:x2] = baseline_value

                    logits = model(occluded)
                    prob = F.softmax(logits, dim=1)[0, target_class].item()
                    drop = base_prob - prob

                    heatmap[t, y:y2, x:x2] += drop
                    counts[t, y:y2, x:x2] += 1

    counts[counts == 0] = 1
    heatmap = heatmap / counts
    heatmap = np.clip(heatmap, a_min=0, a_max=None)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap, target_class
