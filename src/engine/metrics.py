"""
Reusable classification-metrics computation.

Built for (freeze + fully document the baseline), and designed to be
reused unchanged later for:
  - (standardized evaluation across all four architectures)
  - (EF threshold sensitivity sweep, same function, different
    `threshold` argument, no retraining needed since it works off saved
    probabilities)
  - (error-vs-EF / grey zone analysis, uses the same per-sample
    predictions this module is fed)

Label convention in this project (see src/data/dataset.py):
    label = 1  ->  Normal    (EF >= ef_threshold)
    label = 0  ->  Abnormal  (EF <  ef_threshold)

Note: class 1 is Normal in this project, while clinically the "positive" class usually refers to disease/Abnormal. 
sklearn therefore reports its default class-1 metrics for Normal, so I also report the corresponding Abnormal metrics explicitly.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    precision_recall_curve,
)


def _to_native(value):
    """Convert NumPy scalar values to plain Python types for JSON serialization."""
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> Tuple[Dict, np.ndarray]:
    """
    y_true: array-like of {0,1} ground-truth labels (1 = Normal, 0 = Abnormal)
    y_prob: array-like of predicted probability of class 1 (Normal)
    threshold: probability cutoff used to turn y_prob into a hard prediction

    Returns (metrics_dict, confusion_matrix) where confusion_matrix is a
    2x2 numpy array with rows/cols ordered [0 (Abnormal), 1 (Normal)]:

        cm[0,0] = TN  (true Abnormal, predicted Abnormal)
        cm[0,1] = FP  (true Abnormal, predicted Normal)
        cm[1,0] = FN  (true Normal,   predicted Abnormal)
        cm[1,1] = TP  (true Normal,   predicted Normal)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Keep the class order fixed so index 0 = Abnormal and index 1 = Normal.
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        # happens if a split ends up single-class only
        auc = float("nan")

    metrics = {
        "threshold": threshold,
        "n_samples": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": auc,

        # sklearn-default framing (positive class = label 1 = Normal)
        "precision_class1_normal": precision[1],
        "recall_class1_normal": recall[1],
        "f1_class1_normal": f1_per_class[1],

        # Metrics for Abnormal (class 0)
        "precision_class0_abnormal": precision[0],
        "recall_class0_abnormal": recall[0],
        "f1_class0_abnormal": f1_per_class[0],

        # Clinical framing (positive = disease/Abnormal). Sensitivity here means
        # sensitivity for detecting Abnormal, specificity means correctly
        # clearing true Normal cases.
        "sensitivity_abnormal": recall[0],
        "specificity_abnormal": recall[1],

        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    metrics = {k: _to_native(v) for k, v in metrics.items()}
    return metrics, cm


def compute_roc_curve(y_true, y_prob):
    """Returns (fpr, tpr, thresholds). Used in Phase 3 for the ROC plot."""
    return roc_curve(np.asarray(y_true).astype(int), np.asarray(y_prob))


def compute_pr_curve(y_true, y_prob):
    """Returns (precision, recall, thresholds). Used in Phase 3 for the PR plot."""
    return precision_recall_curve(np.asarray(y_true).astype(int), np.asarray(y_prob))