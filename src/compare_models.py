"""
Phase 3: compare the main architectures on the same evaluation outputs. 
Reads the *_predictions.csv files saved by src/evaluate.py for each model, 
so no new inference or GPU is needed here. 

The script overlays the models on combined ROC and Precision-Recall plots using the same plotting functions as 
the individual model evaluation. It also creates a combined metrics table with Accuracy, 
Precision, Recall, F1, Sensitivity, Specificity, and ROC-AUC for each model. 

Usage (run once all four *_predictions.csv exist in results/evaluation/):
    python -m src.compare_models --eval_dir results/evaluation --split test \
        --out_dir results/evaluation
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.engine.metrics import compute_metrics
from src.engine.plots import plot_roc_curve, plot_pr_curve


# run_name prefix -> display label, in the order they should appear in plots/tables
MODEL_DISPLAY_NAMES = {
    "baseline": "Baseline (CNN, avg.)",
    "cnn_lstm": "CNN + LSTM",
    "r3d": "R3D-18",
    "swin3d": "Swin3D-T",
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", type=str, default="results/evaluation",
                    help="directory containing the *_predictions.csv files from src/evaluate.py")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--classification_threshold", type=float, default=0.5)
    p.add_argument("--out_dir", type=str, default="results/evaluation")
    return p.parse_args()


def find_predictions_file(eval_dir, model_key, split):
    pattern = os.path.join(eval_dir, f"{model_key}*_{split}_predictions.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[warn] multiple predictions files match '{model_key}*_{split}_predictions.csv', "
              f"using the most recent: {matches[-1]}")
    return matches[-1]


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    loaded = {}
    for model_key in MODEL_DISPLAY_NAMES:
        path = find_predictions_file(args.eval_dir, model_key, args.split)
        if path is None:
            print(f"[skip] no predictions file found for '{model_key}' in {args.eval_dir} "
                  f"(expected something like {model_key}*_{args.split}_predictions.csv - "
                  "run src/evaluate.py for this model first)")
            continue
        df = pd.read_csv(path)
        loaded[model_key] = df
        print(f"[loaded] {model_key}: {path} ({len(df)} rows)")

    if not loaded:
        print("Nothing to compare - no predictions files found. Run src/evaluate.py "
              "for at least two models first.")
        return

    # Combined ROC plot 
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for model_key, df in loaded.items():
        plot_roc_curve(df["true_label"], df["pred_prob"], save_path=None,
                        label=MODEL_DISPLAY_NAMES[model_key], ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Architectures")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = os.path.join(args.out_dir, f"comparison_{args.split}_roc_curves.png")
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved combined ROC comparison to: {roc_path}")

    # Combined PR plot 
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for model_key, df in loaded.items():
        plot_pr_curve(df["true_label"], df["pred_prob"], save_path=None,
                      label=MODEL_DISPLAY_NAMES[model_key], ax=ax)
    overall_baseline = float(np.mean(np.concatenate([df["true_label"].values for df in loaded.values()])))
    ax.axhline(overall_baseline, linestyle="--", color="gray", linewidth=1,
               label=f"Chance (~{overall_baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves - All Architectures")
    ax.legend(loc="lower left")
    fig.tight_layout()
    pr_path = os.path.join(args.out_dir, f"comparison_{args.split}_pr_curves.png")
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    print(f"Saved combined PR comparison to: {pr_path}")

    # Combined metrics table (the "main model comparison" table)
    rows = []
    for model_key, df in loaded.items():
        metrics, _ = compute_metrics(df["true_label"], df["pred_prob"],
                                      threshold=args.classification_threshold)
        rows.append({
            "model": MODEL_DISPLAY_NAMES[model_key],
            "accuracy": round(metrics["accuracy"], 4),
            "precision_normal": round(metrics["precision_class1_normal"], 4),
            "recall_normal": round(metrics["recall_class1_normal"], 4),
            "f1_normal": round(metrics["f1_class1_normal"], 4),
            "sensitivity_abnormal": round(metrics["sensitivity_abnormal"], 4),
            "specificity_abnormal": round(metrics["specificity_abnormal"], 4),
            "roc_auc": round(metrics["roc_auc"], 4),
        })
    table_df = pd.DataFrame(rows)
    table_path = os.path.join(args.out_dir, f"comparison_{args.split}_summary_table.csv")
    table_df.to_csv(table_path, index=False)
    print(f"\nCombined comparison table:\n{table_df.to_string(index=False)}")
    print(f"\nSaved combined summary table to: {table_path}")


if __name__ == "__main__":
    main()
