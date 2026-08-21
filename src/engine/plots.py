"""Reusable plotting functions for the complete evaluation outputs in 
Phase 3: ROC curve, Precision-Recall curve, and confusion matrix heatmap. 
These functions take the arrays/values directly, so I can use the same 
plotting code when evaluating one model in src/evaluate.py and when comparing 
all architectures together in src/compare_models.py. This keeps the plotting 
logic in one place instead of implementing the same plots separately for evaluation 
and model comparison."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.engine.metrics import compute_roc_curve, compute_pr_curve


def plot_roc_curve(y_true, y_prob, save_path, title=None, label=None, ax=None):
    """Plot the ROC curve for one model. Normally this creates and saves its own figure. 
    If an \ax` is passed, the curve is drawn on that axis instead, which is how compare_models.py 
    puts several model curves on the same plot."""
    fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
    from sklearn.metrics import auc as sk_auc
    roc_auc = sk_auc(fpr, tpr)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    curve_label = label or f"AUC = {roc_auc:.3f}"
    if label is not None:
        curve_label = f"{label} (AUC = {roc_auc:.3f})"
    ax.plot(fpr, tpr, linewidth=2, label=curve_label)

    if standalone:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title or "ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return roc_auc


def plot_pr_curve(y_true, y_prob, save_path, title=None, label=None, ax=None):
    """Plot the Precision-Recall curve for one model. 
    As with the ROC function, passing an \ax` lets me draw several model curves on the same figure. 
    The positive class here is label 1 (Normal), which is the same convention used in src/engine/metrics.py."""
    precision, recall, _ = compute_pr_curve(y_true, y_prob)
    from sklearn.metrics import auc as sk_auc
    pr_auc = sk_auc(recall, precision)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    curve_label = label or f"AP = {pr_auc:.3f}"
    if label is not None:
        curve_label = f"{label} (AP = {pr_auc:.3f})"
    ax.plot(recall, precision, linewidth=2, label=curve_label)

    if standalone:
        baseline = float(np.mean(y_true))
        ax.axhline(baseline, linestyle="--", color="gray", linewidth=1,
                    label=f"Chance ({baseline:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title or "Precision-Recall Curve")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return pr_auc


def plot_confusion_matrix(cm, save_path, title=None,
                           class_names=("Abnormal", "Normal")):
    """2x2 confusion matrix heatmap with counts annotated in each cell."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred {c}" for c in class_names])
    ax.set_yticklabels([f"True {c}" for c in class_names])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14)

    ax.set_title(title or "Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
