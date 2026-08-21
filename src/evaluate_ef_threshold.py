"""
Phase 7: EF threshold sensitivity sweep. 
Recomputes the ground-truth labels at several EF thresholds from the 
*_predictions.csv files saved by src.evaluate.py. No retraining or 
new inference is done here: each model keeps the same predicted 
probability produced by the model trained with EF=50 as the classification boundary. 

For each candidate EF threshold, only the ground-truth Normal/Abnormal labels are changed. 
This tests how well the existing model predictions separate different EF cutoffs, 
which is different from retraining a new model for each threshold. 
The classification decision itself stays fixed at pred_prob >= 0.5 -> Normal. 

Usage (single model):
    python -m src.evaluate_ef_threshold \
        --predictions_csv results/evaluation/r3d_64f_112px_clipp1_pretrained_best_test_predictions.csv \
        --run_name r3d --out_dir results/ef_threshold

Usage (all four models at once, auto-discovering predictions files the same
way src/compare_models.py does):
    python -m src.evaluate_ef_threshold --eval_dir results/evaluation --out_dir results/ef_threshold
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.engine.metrics import compute_metrics

DEFAULT_THRESHOLDS = [40, 45, 50, 55, 60, 65, 70]

MODEL_DISPLAY_NAMES = {
    "baseline": "Baseline (CNN, avg.)",
    "cnn_lstm": "CNN + LSTM",
    "r3d": "R3D-18",
    "swin3d": "Swin3D-T",
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_csv", type=str, default=None,
                    help="run for a single model's *_predictions.csv")
    p.add_argument("--run_name", type=str, default=None,
                    help="label for the single-model run (required if --predictions_csv is given)")
    p.add_argument("--eval_dir", type=str, default="results/evaluation",
                    help="if --predictions_csv is not given, auto-discover all four models' "
                         "*_predictions.csv here instead (same convention as src/compare_models.py)")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    p.add_argument("--classification_threshold", type=float, default=0.5,
                    help="probability cut for the decision rule - fixed as-trained, NOT swept")
    p.add_argument("--out_dir", type=str, default="results/ef_threshold")
    return p.parse_args()


def find_predictions_file(eval_dir, model_key, split):
    pattern = os.path.join(eval_dir, f"{model_key}*_{split}_predictions.csv")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def sweep_one_model(df, run_name, thresholds, classification_threshold, out_dir):
    rows = []
    for ef_thr in thresholds:
        y_true = (df["ef"] >= ef_thr).astype(int)
        n_normal = int(y_true.sum())
        n_abnormal = int(len(y_true) - n_normal)

        metrics, cm = compute_metrics(y_true, df["pred_prob"], threshold=classification_threshold)

        row = {
            "ef_threshold": ef_thr,
            "n_normal": n_normal,
            "n_abnormal": n_abnormal,
            "pct_abnormal": round(100 * n_abnormal / len(y_true), 1),
            **metrics,
        }
        rows.append(row)

    result_df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"{run_name}_ef_threshold_sweep.csv")
    result_df.to_csv(csv_path, index=False)

    cols = ["ef_threshold", "pct_abnormal", "accuracy", "f1_class1_normal", "roc_auc",
            "sensitivity_abnormal", "specificity_abnormal"]
    print(f"\n{run_name}:")
    print(result_df[cols].to_string(index=False))
    print(f"Saved: {csv_path}")

    # per-model plot: EF threshold -> Accuracy / F1 / AUC / Sensitivity / Specificity 
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(result_df["ef_threshold"], result_df["accuracy"], marker="o", label="Accuracy")
    ax.plot(result_df["ef_threshold"], result_df["f1_class1_normal"], marker="o", label="F1 (Normal)")
    ax.plot(result_df["ef_threshold"], result_df["roc_auc"], marker="o", label="ROC-AUC")
    ax.plot(result_df["ef_threshold"], result_df["sensitivity_abnormal"], marker="o",
             linestyle="--", label="Sensitivity (Abnormal)")
    ax.plot(result_df["ef_threshold"], result_df["specificity_abnormal"], marker="o",
             linestyle="--", label="Specificity (Abnormal)")
    ax.axvline(50, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("EF threshold used for ground-truth labels (%)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"EF threshold sensitivity - {run_name}")
    ax.legend(loc="lower center", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, f"{run_name}_ef_threshold_sweep.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {plot_path}")

    return result_df


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}

    if args.predictions_csv:
        if not args.run_name:
            raise ValueError("--run_name is required when using --predictions_csv")
        df = pd.read_csv(args.predictions_csv)
        all_results[args.run_name] = sweep_one_model(
            df, args.run_name, args.thresholds, args.classification_threshold, args.out_dir)
    else:
        for model_key, display_name in MODEL_DISPLAY_NAMES.items():
            path = find_predictions_file(args.eval_dir, model_key, args.split)
            if path is None:
                print(f"[skip] no predictions file found for '{model_key}' in {args.eval_dir}")
                continue
            df = pd.read_csv(path)
            print(f"[loaded] {model_key}: {path} ({len(df)} rows)")
            all_results[display_name] = sweep_one_model(
                df, model_key, args.thresholds, args.classification_threshold, args.out_dir)

    if len(all_results) < 2:
        return

    # combined AUC vs threshold plot across all models
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, df in all_results.items():
        ax.plot(df["ef_threshold"], df["roc_auc"], marker="o", label=label)
    ax.axvline(50, color="gray", linestyle=":", linewidth=1, label="Threshold used for training (50%)")
    ax.set_xlabel("EF threshold used for ground-truth labels (%)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC across EF thresholds - all architectures")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    combined_auc_path = os.path.join(args.out_dir, "comparison_ef_threshold_auc.png")
    fig.savefig(combined_auc_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved combined AUC-vs-threshold plot to: {combined_auc_path}")

    # combined F1-vs-threshold plot across all models
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, df in all_results.items():
        ax.plot(df["ef_threshold"], df["f1_class1_normal"], marker="o", label=label)
    ax.axvline(50, color="gray", linestyle=":", linewidth=1, label="Threshold used for training (50%)")
    ax.set_xlabel("EF threshold used for ground-truth labels (%)")
    ax.set_ylabel("F1 (Normal)")
    ax.set_title("F1 across EF thresholds - all architectures")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    combined_f1_path = os.path.join(args.out_dir, "comparison_ef_threshold_f1.png")
    fig.savefig(combined_f1_path, dpi=150)
    plt.close(fig)
    print(f"Saved combined F1-vs-threshold plot to: {combined_f1_path}")


if __name__ == "__main__":
    main()