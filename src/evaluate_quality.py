"""
Phase 6: evaluate model robustness to changes in echocardiographic image quality. 
Loads an already trained checkpoint and evaluates the same test set at four 
controlled quality levels: clean, mild, moderate, and severe (see src/data/degradation.py). 

No retraining is done. The clean condition should reproduce the Phase 1/3 evaluation 
for the same checkpoint, apart from the deterministic evaluation sampling already used there, 
so it also acts as a check that the quality pipeline itself has not changed the normal evaluation setup.

Usage:
    python -m src.evaluate_quality --model baseline \
        --checkpoint checkpoints/baseline_64f_112px_clipp1_pretrained_best.pt \
        --num_frames 64 --img_size 112 --temporal_sampling clip --clip_period 1 \
        --out_dir results/quality_robustness
"""

import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import EchoVideoDataset, make_splits
from src.data.degradation import QUALITY_LEVELS
from src.models import MODEL_REGISTRY, MODEL_FAMILY, MODEL_NORMALIZE
from src.engine.common import predict_all
from src.engine.metrics import compute_metrics


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    p.add_argument("--checkpoint", type=str, required=True)

    p.add_argument("--data_root", type=str,
                    default=os.environ.get("ECHO_DATA_ROOT", "/scratch/project_2018481/echonet"))

    p.add_argument("--num_frames", type=int, required=True)
    p.add_argument("--img_size", type=int, required=True)
    p.add_argument("--temporal_sampling", choices=["uniform", "clip"], default="clip")
    p.add_argument("--clip_period", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--ef_threshold", type=float, default=50.0)
    p.add_argument("--classification_threshold", type=float, default=0.5)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dropout", type=float, default=0.3)

    p.add_argument("--out_dir", type=str, default="results/quality_robustness")
    p.add_argument("--run_name", type=str, default=None,
                    help="defaults to the checkpoint filename without extension")
    return p.parse_args()


def build_model_for_eval(args):
    model_cls = MODEL_REGISTRY[args.model]
    kwargs = dict(pretrained=False, dropout=args.dropout, num_classes=2)
    if args.model == "cnn_lstm":
        kwargs.update(hidden_size=256, num_layers=1, bidirectional=False)
    return model_cls(**kwargs)


def main():
    args = get_args()
    run_name = args.run_name or os.path.splitext(os.path.basename(args.checkpoint))[0]
    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.data_root, "FileList.csv")
    video_dir = os.path.join(args.data_root, "Videos")
    df = pd.read_csv(csv_path)
    train_df, val_df, test_df = make_splits(df)
    split_df = {"val": val_df, "test": test_df}[args.split]

    model_family = MODEL_FAMILY[args.model]
    normalize = MODEL_NORMALIZE[args.model]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_model_for_eval(args).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    for level in QUALITY_LEVELS:
        dataset = EchoVideoDataset(
            df=split_df,
            video_dir=video_dir,
            num_frames=args.num_frames,
            img_size=args.img_size,
            ef_threshold=args.ef_threshold,
            model_family=model_family,
            normalize=normalize,
            temporal_sampling=args.temporal_sampling,
            clip_period=args.clip_period,
            split=args.split,
            seed=args.seed,
            quality_level=level,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        )

        print(f"\n[{level}] running inference on {len(dataset)} '{args.split}' videos...")
        preds = predict_all(model, loader, device)
        metrics, cm = compute_metrics(preds["label"], preds["prob"],
                                       threshold=args.classification_threshold)
        metrics["quality_level"] = level
        rows.append(metrics)
        print(f"[{level}] Acc={metrics['accuracy']:.4f}  F1={metrics['f1_class1_normal']:.4f}  "
              f"AUC={metrics['roc_auc']:.4f}  Sensitivity(abnormal)={metrics['sensitivity_abnormal']:.4f}  "
              f"Specificity(abnormal)={metrics['specificity_abnormal']:.4f}")

    results_df = pd.DataFrame(rows)
    cols = ["quality_level", "accuracy", "f1_class1_normal", "roc_auc",
            "sensitivity_abnormal", "specificity_abnormal", "precision_class1_normal",
            "recall_class1_normal"]
    results_df = results_df[cols + [c for c in results_df.columns if c not in cols]]

    csv_path_out = os.path.join(args.out_dir, f"{run_name}_{args.split}_quality_sweep.csv")
    results_df.to_csv(csv_path_out, index=False)
    print(f"\nSaved quality sweep table to: {csv_path_out}")
    print(results_df[cols].to_string(index=False))

    # degradation curve plot
    levels = list(QUALITY_LEVELS.keys())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(levels, results_df["accuracy"], marker="o", label="Accuracy")
    ax.plot(levels, results_df["f1_class1_normal"], marker="o", label="F1 (Normal)")
    ax.plot(levels, results_df["roc_auc"], marker="o", label="ROC-AUC")
    ax.plot(levels, results_df["sensitivity_abnormal"], marker="o", linestyle="--",
             label="Sensitivity (Abnormal)")
    ax.set_xlabel("Quality level (increasing degradation ->)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Quality robustness - {run_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(args.out_dir, f"{run_name}_{args.split}_quality_sweep.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved quality sweep plot to: {plot_path}")

    # save the frozen config for reference
    config_path = os.path.join(args.out_dir, f"{run_name}_{args.split}_quality_sweep_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved config to: {config_path}")


if __name__ == "__main__":
    main()