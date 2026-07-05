"""
Generates interpretability overlays for a handful of test videos, including
ones near the EF=50 decision boundary (the cases my supervisor is most
interested in for error analysis).

Usage example:
    python -m src.interpret.run_interpretability \
        --model r3d --checkpoint checkpoints/r3d18_64f_best.pt \
        --num_frames 64 --img_size 112 --temporal_sampling clip --clip_period 1 \
        --num_examples 6 --out_dir results/interpretability

For Swin3D, Grad-CAM isn't applicable (no conv layer to hook), so this script
automatically falls back to occlusion sensitivity for that model.
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch

from src.models import MODEL_REGISTRY, MODEL_FAMILY, MODEL_NORMALIZE
from src.data.dataset import EchoVideoDataset, make_splits
from src.interpret.grad_cam import (
    compute_gradcam_r3d,
    compute_gradcam_2d_backbone,
    occlusion_sensitivity,
)
from src.data.video_utils import get_num_frames, uniform_indices, clip_indices, read_video_frames


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, default=os.environ.get(
        "ECHO_DATA_ROOT", "/scratch/project_2018481/relbouaz/echonet"))
    p.add_argument("--num_frames", type=int, default=64)
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--temporal_sampling", choices=["uniform", "clip"], default="clip")
    p.add_argument("--clip_period", type=int, default=1)
    p.add_argument("--ef_threshold", type=float, default=50.0)
    p.add_argument("--num_examples", type=int, default=6)
    p.add_argument("--near_boundary_width", type=float, default=5.0,
                    help="pick examples with EF within this many points of the threshold")
    p.add_argument("--out_dir", type=str, default="results/interpretability")
    p.add_argument("--method", choices=["auto", "gradcam", "occlusion"], default="auto")
    return p.parse_args()


def overlay_cam_on_frame(frame_rgb: np.ndarray, cam_2d: np.ndarray) -> np.ndarray:
    """frame_rgb in [0,255] uint8 HxWx3, cam_2d in [0,1] HxW."""
    heatmap = cv2.applyColorMap((cam_2d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * frame_rgb + 0.45 * heatmap).astype(np.uint8)
    return overlay


def pick_examples(test_df: pd.DataFrame, ef_threshold: float, width: float, n: int) -> pd.DataFrame:
    near = test_df[(test_df["EF"] - ef_threshold).abs() <= width]
    if len(near) >= n:
        return near.sample(n=n, random_state=42)
    # not enough near-boundary cases, fill the rest with a random spread
    remaining = n - len(near)
    rest = test_df.drop(near.index).sample(n=min(remaining, len(test_df) - len(near)), random_state=42)
    return pd.concat([near, rest])


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.data_root, "FileList.csv")
    video_dir = os.path.join(args.data_root, "Videos")
    df = pd.read_csv(csv_path)
    _, _, test_df = make_splits(df)

    examples = pick_examples(test_df, args.ef_threshold, args.near_boundary_width, args.num_examples)
    print(f"Selected {len(examples)} example videos (near-boundary width={args.near_boundary_width}):")
    print(examples[["FileName", "EF"]].to_string(index=False))

    model_cls = MODEL_REGISTRY[args.model]
    model_family = MODEL_FAMILY[args.model]
    normalize = MODEL_NORMALIZE[args.model]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls(pretrained=False)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    method = args.method
    if method == "auto":
        method = "occlusion" if args.model == "swin3d" else "gradcam"
    print(f"Using interpretability method: {method}")

    ds = EchoVideoDataset(
        df=examples,
        video_dir=video_dir,
        num_frames=args.num_frames,
        img_size=args.img_size,
        ef_threshold=args.ef_threshold,
        model_family=model_family,
        normalize=normalize,
        temporal_sampling=args.temporal_sampling,
        clip_period=args.clip_period,
        split="test",
    )

    for i in range(len(ds)):
        sample = ds[i]
        video = sample["video"].unsqueeze(0).to(device)
        label = sample["label"].item()
        ef = sample["ef"].item()
        filename = sample["filename"]

        video.requires_grad_(True)

        if method == "gradcam" and args.model == "r3d":
            cam, pred_class = compute_gradcam_r3d(model, video)
            # cam has fewer temporal bins than num_frames (R3D downsamples
            # time); repeat each bin to line back up with the sampled frames
            # for visualization purposes.
            reps = int(np.ceil(args.num_frames / cam.shape[0]))
            cam_full = np.repeat(cam, reps, axis=0)[: args.num_frames]
        elif method == "gradcam" and args.model in ("baseline", "cnn_lstm"):
            cam_full, pred_class = compute_gradcam_2d_backbone(model, video)
        else:
            cam_full, pred_class = occlusion_sensitivity(model, video, patch_size=16, stride=16)

        # Re-decode the exact raw (un-normalized) frames the dataset sampled,
        # by re-running the same deterministic sampler used for the test
        # split, so the overlay is drawn on real pixel values instead of a
        # normalized tensor.
        video_path = os.path.join(video_dir, f"{filename}.avi")

        example_dir = os.path.join(args.out_dir, f"{args.model}_{filename}_EF{ef:.0f}_label{label}")
        os.makedirs(example_dir, exist_ok=True)
        total_frames = get_num_frames(video_path)
        if args.temporal_sampling == "uniform":
            indices = uniform_indices(total_frames, args.num_frames)
        else:
            indices = clip_indices(total_frames, args.num_frames, period=args.clip_period, random_start=False)
        raw_video = read_video_frames(video_path, indices, img_size=args.img_size)  # (T,C,H,W) in [0,1]
        raw_video = (raw_video.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)  # (T,H,W,C)

        n_save = min(8, raw_video.shape[0])
        save_indices = np.linspace(0, raw_video.shape[0] - 1, n_save).astype(int)
        for t in save_indices:
            overlay = overlay_cam_on_frame(raw_video[t], cam_full[t])
            out_path = os.path.join(example_dir, f"frame_{t:03d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(
            f"[{filename}] true_label={label} (EF={ef:.1f}) pred_class={pred_class} "
            f"-> saved {n_save} overlay frames to {example_dir}"
        )

    print(f"\nDone. Overlays saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
