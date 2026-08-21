"""
Phase 8: Error analysis and the EF "grey zone". 
This uses the *_predictions.csv files already produced by src/evaluate.py, 
so no new inference is needed. The aim is to check whether classification 
errors are concentrated around the EF=50 decision boundary. 

I look at the errors in two ways: (a) error rate across absolute EF bins, 
and (b) error rate across bins of distance from the threshold, |EF-50|. 
The second view is useful for testing the grey-zone idea directly, because 
it ignores which side of the boundary the sample is on and only measures how close it is to 50. 

I also plot predicted probability against EF so that cases close to both EF=50 and pred_prob=0.5 
can be inspected directly. Usage (all four models):
    python -m src.analyze_grey_zone --eval_dir results/evaluation --out_dir results/grey_zone
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

EF_THRESHOLD = 50.0
DISTANCE_BINS = [0, 5, 10, 15, 20, 30, 100]
DISTANCE_LABELS = ["0-5", "5-10", "10-15", "15-20", "20-30", "30+"]

MODEL_DISPLAY_NAMES = {
    "baseline": "Baseline (CNN, avg.)",
    "cnn_lstm": "CNN + LSTM",
    "r3d": "R3D-18",
    "swin3d": "Swin3D-T",
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_csv", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--eval_dir", type=str, default="results/evaluation")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--out_dir", type=str, default="results/grey_zone")
    return p.parse_args()


def find_predictions_file(eval_dir, model_key, split):
    pattern = os.path.join(eval_dir, f"{model_key}*_{split}_predictions.csv")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def analyze_one_model(df: pd.DataFrame, run_name: str, out_dir: str) -> pd.DataFrame:
    df = df.copy()
    if "pred_label" not in df.columns:
        df["pred_label"] = (df["pred_prob"] >= 0.5).astype(int)
    df["correct"] = df["pred_label"] == df["true_label"]
    df["distance_from_threshold"] = (df["ef"] - EF_THRESHOLD).abs()
    # under-call: model says Normal (1) but truth is Abnormal (0) - a missed case
    df["under_call"] = (df["true_label"] == 0) & (df["pred_label"] == 1)
    # over-call: model says Abnormal (0) but truth is Normal (1) - a false alarm
    df["over_call"] = (df["true_label"] == 1) & (df["pred_label"] == 0)

    # (a) error rate by absolute EF bin 
    ef_min, ef_max = np.floor(df["ef"].min() / 5) * 5, np.ceil(df["ef"].max() / 5) * 5
    ef_bins = np.arange(ef_min, ef_max + 5, 5)
    df["ef_bin"] = pd.cut(df["ef"], bins=ef_bins, right=False)
    ef_bin_table = df.groupby("ef_bin", observed=True).agg(
        n=("correct", "size"),
        error_rate=("correct", lambda s: 1 - s.mean()),
        under_call_rate=("under_call", "mean"),
        over_call_rate=("over_call", "mean"),
    ).reset_index()
    ef_bin_table["ef_bin"] = ef_bin_table["ef_bin"].astype(str)
    ef_bin_path = os.path.join(out_dir, f"{run_name}_error_by_ef_bin.csv")
    ef_bin_table.to_csv(ef_bin_path, index=False)

    # (b) error rate by distance from the 50% boundary
    df["distance_bin"] = pd.cut(df["distance_from_threshold"], bins=DISTANCE_BINS,
                                  labels=DISTANCE_LABELS, right=False)
    dist_bin_table = df.groupby("distance_bin", observed=True).agg(
        n=("correct", "size"),
        error_rate=("correct", lambda s: 1 - s.mean()),
        under_call_rate=("under_call", "mean"),
        over_call_rate=("over_call", "mean"),
    ).reindex(DISTANCE_LABELS).reset_index()
    dist_bin_path = os.path.join(out_dir, f"{run_name}_error_by_distance_bin.csv")
    dist_bin_table.to_csv(dist_bin_path, index=False)

    print(f"\n{run_name} - error rate by distance from 50% boundary:")
    print(dist_bin_table.to_string(index=False))

    # statistical test: is |EF-50| different for correct vs incorrect predictions? 
    correct_dist = df.loc[df["correct"], "distance_from_threshold"]
    incorrect_dist = df.loc[~df["correct"], "distance_from_threshold"]
    u_stat, p_value = scipy_stats.mannwhitneyu(incorrect_dist, correct_dist, alternative="less")
    print(f"{run_name}: median |EF-50| for correct={correct_dist.median():.2f}, "
          f"incorrect={incorrect_dist.median():.2f}  "
          f"(Mann-Whitney U, H1: errors are closer to boundary: p={p_value:.2e})")

    # plot A: predicted probability vs EF, colored by correctness
    fig, ax = plt.subplots(figsize=(8, 5.5))
    correct_pts = df[df["correct"]]
    wrong_pts = df[~df["correct"]]
    ax.scatter(correct_pts["ef"], correct_pts["pred_prob"], s=12, alpha=0.4,
               color="tab:blue", label=f"Correct (n={len(correct_pts)})")
    ax.scatter(wrong_pts["ef"], wrong_pts["pred_prob"], s=18, alpha=0.8,
               color="tab:red", label=f"Incorrect (n={len(wrong_pts)})", marker="x")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.axvline(EF_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Ground-truth EF (%)")
    ax.set_ylabel("Predicted probability of Normal")
    ax.set_title(f"Prediction confidence vs. EF - {run_name}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    scatter_path = os.path.join(out_dir, f"{run_name}_confidence_vs_ef_scatter.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    # plot B: error rate by distance-from-boundary bin
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(DISTANCE_LABELS))
    ax1.bar(x, dist_bin_table["error_rate"], color="tab:red", alpha=0.7, label="Error rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(DISTANCE_LABELS)
    ax1.set_xlabel("Distance from 50% EF boundary")
    ax1.set_ylabel("Error rate")
    ax1.set_ylim(0, max(0.5, dist_bin_table["error_rate"].max() * 1.2))
    ax2 = ax1.twinx()
    ax2.plot(x, dist_bin_table["n"], color="black", marker="o", linewidth=1, label="n samples")
    ax2.set_ylabel("n samples")
    ax1.set_title(f"Error rate vs. distance from EF boundary - {run_name}")
    fig.tight_layout()
    dist_plot_path = os.path.join(out_dir, f"{run_name}_error_by_distance_bin.png")
    fig.savefig(dist_plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {ef_bin_path}, {dist_bin_path}, {scatter_path}, {dist_plot_path}")

    dist_bin_table["run_name"] = run_name
    return dist_bin_table


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_dist_tables = {}

    if args.predictions_csv:
        if not args.run_name:
            raise ValueError("--run_name is required when using --predictions_csv")
        df = pd.read_csv(args.predictions_csv)
        all_dist_tables[args.run_name] = analyze_one_model(df, args.run_name, args.out_dir)
    else:
        for model_key, display_name in MODEL_DISPLAY_NAMES.items():
            path = find_predictions_file(args.eval_dir, model_key, args.split)
            if path is None:
                print(f"[skip] no predictions file found for '{model_key}' in {args.eval_dir}")
                continue
            df = pd.read_csv(path)
            print(f"\n[loaded] {model_key}: {path} ({len(df)} rows)")
            all_dist_tables[display_name] = analyze_one_model(df, model_key, args.out_dir)

    if len(all_dist_tables) < 2:
        return

    # combined plot: error rate vs distance-from-boundary, all models
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(DISTANCE_LABELS))
    for label, table in all_dist_tables.items():
        ax.plot(x, table["error_rate"], marker="o", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(DISTANCE_LABELS)
    ax.set_xlabel("Distance from 50% EF boundary")
    ax.set_ylabel("Error rate")
    ax.set_title("Error rate vs. distance from EF boundary - all architectures")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    combined_path = os.path.join(args.out_dir, "comparison_error_by_distance_bin.png")
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved combined plot to: {combined_path}")


if __name__ == "__main__":
    main()