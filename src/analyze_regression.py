"""
Phase 9: analyze EF regression results. 

Reads the per-sample predictions saved by src/train_regression.py (filename, true_ef, pred_ef, abs_error)
and produces: true EF vs predicted EF / true EF vs absolute prediction error 

It also checks whether regression error changes with distance from the 50% EF boundary, 
to compare with the grey-zone analysis from Phase 8. 

Finally, I threshold the predicted EF at 50% to derive a binary classifier and compare its classification
 metrics with the directly trained classifiers from Phase 1/3. 
 
 Usage:
    python -m src.analyze_regression \
        --predictions_csv results/r3d_regression_64f_112px_clipp1_pretrained_test_predictions.csv \
        --out_dir results/regression
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.engine.metrics import compute_metrics

EF_THRESHOLD = 50.0


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="results/regression")
    p.add_argument("--run_name", type=str, default="r3d_regression")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.predictions_csv)
    if "abs_error" not in df.columns:
        df["abs_error"] = (df["pred_ef"] - df["true_ef"]).abs()

    mae = df["abs_error"].mean()
    rmse = np.sqrt((df["pred_ef"] - df["true_ef"]).pow(2).mean())
    r2 = 1 - ((df["pred_ef"] - df["true_ef"]) ** 2).sum() / ((df["true_ef"] - df["true_ef"].mean()) ** 2).sum()
    print(f"Overall: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  (n={len(df)})")

    # plot 1: true vs predicted EF 
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(df["true_ef"], df["pred_ef"], s=14, alpha=0.4, color="tab:blue")
    lims = [min(df["true_ef"].min(), df["pred_ef"].min()) - 2,
            max(df["true_ef"].max(), df["pred_ef"].max()) + 2]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="y = x (perfect prediction)")
    ax.axvline(EF_THRESHOLD, color="crimson", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(EF_THRESHOLD, color="crimson", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True EF (%)")
    ax.set_ylabel("Predicted EF (%)")
    ax.set_title(f"True vs. Predicted EF (MAE={mae:.2f}, R2={r2:.3f})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path1 = os.path.join(args.out_dir, f"{args.run_name}_true_vs_pred_ef.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"Saved: {path1}")

    # plot 2: absolute error vs true EF 
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(df["true_ef"], df["abs_error"], s=14, alpha=0.4, color="tab:orange")
    ax.axvline(EF_THRESHOLD, color="crimson", linestyle=":", linewidth=1, alpha=0.6,
               label="50% boundary")
    # rolling mean trend line for readability
    order = df["true_ef"].argsort()
    sorted_ef = df["true_ef"].values[order]
    sorted_err = df["abs_error"].values[order]
    window = max(5, len(df) // 30)
    if len(df) > window:
        rolling = pd.Series(sorted_err).rolling(window, center=True, min_periods=1).mean()
        ax.plot(sorted_ef, rolling, color="black", linewidth=2, label=f"rolling mean (w={window})")
    ax.set_xlabel("True EF (%)")
    ax.set_ylabel("Absolute error (EF percentage points)")
    ax.set_title("Prediction error vs. true EF")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path2 = os.path.join(args.out_dir, f"{args.run_name}_abs_error_vs_ef.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"Saved: {path2}")

    # connect to Phase 8: error vs distance from 50% boundary 
    df["distance_from_threshold"] = (df["true_ef"] - EF_THRESHOLD).abs()
    corr, p_value = scipy_stats.spearmanr(df["distance_from_threshold"], df["abs_error"])
    print(f"\nSpearman correlation between |EF-50| and |prediction error|: "
          f"rho={corr:.3f}, p={p_value:.2e}")
    if p_value < 0.05 and corr < 0:
        print("Significant negative correlation: prediction error is higher closer to the "
              "50% boundary - consistent with the classification grey zone found in Phase 8.")
    elif p_value < 0.05 and corr > 0:
        print("Significant positive correlation: prediction error is LOWER closer to the "
              "50% boundary - opposite of the classification grey zone pattern.")
    else:
        print(f"Not statistically significant (p={p_value:.3f} >= 0.05): regression error "
              "does NOT show the same tight grey-zone concentration the classifiers' errors "
              "showed in Phase 8, regardless of the sign of the correlation. This itself is "
              "worth reporting - it suggests the classifiers' grey zone may be partly an "
              "artifact of forcing a hard binary decision near an ambiguous boundary, rather "
              "than a fundamental limit of what the video contains.")

    # derived classifier: threshold the continuous prediction at 50% 
    y_true = (df["true_ef"] >= EF_THRESHOLD).astype(int)
    # Map predicted EF to a probability like score centered at 50 so it 
    # can be passed to compute_metrics with threshold=0.5. This monotonic 
    # transform does not change ROC-AUC because the ranking of predicted EF values stays the same
    pred_prob_like = 1 / (1 + np.exp(-(df["pred_ef"] - EF_THRESHOLD) / 5.0))
    metrics, cm = compute_metrics(y_true, pred_prob_like, threshold=0.5)
    print(f"\nDerived classifier (predicted EF >= 50 -> Normal), compare against Phase 1/3's "
          f"trained R3D-18 classifier (Acc=0.874, F1=0.917, AUC=0.934):")
    print(f"  Accuracy={metrics['accuracy']:.4f}  F1={metrics['f1_class1_normal']:.4f}  "
          f"AUC={metrics['roc_auc']:.4f}  Sensitivity(abnormal)={metrics['sensitivity_abnormal']:.4f}  "
          f"Specificity(abnormal)={metrics['specificity_abnormal']:.4f}")

    summary = {
        "mae": mae, "rmse": rmse, "r2": r2,
        "spearman_corr_distance_vs_error": corr, "spearman_p": p_value,
        "derived_accuracy": metrics["accuracy"], "derived_f1": metrics["f1_class1_normal"],
        "derived_auc": metrics["roc_auc"], "derived_sensitivity_abnormal": metrics["sensitivity_abnormal"],
        "derived_specificity_abnormal": metrics["specificity_abnormal"],
    }
    summary_path = os.path.join(args.out_dir, f"{args.run_name}_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()