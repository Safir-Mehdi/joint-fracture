"""
Classification metrics for fracture detection.

Convention:
    Class 0 = fractured  (POSITIVE — clinical priority)
    Class 1 = normal     (negative)

Usage:
    from src.evaluation.metrics import compute_metrics

    metrics = compute_metrics(y_true, y_pred, y_prob_fractured)
    # → {"accuracy": 0.91, "f1": 0.88, "auc": 0.94, "sensitivity": 0.93, ...}
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Class convention ───────────────────────────────────────
POSITIVE_CLASS = 0         # fractured
CLASS_NAMES = {0: "fractured", 1: "normal"}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    positive_class: int = POSITIVE_CLASS,
) -> dict:
    """
    Compute all classification metrics for binary fracture detection.

    Args:
        y_true:         (N,) ground-truth labels {0, 1}
        y_pred:         (N,) predicted labels    {0, 1}
        y_prob:         (N,) probability of the POSITIVE class (fractured)
        positive_class: which class is "positive" — default 0 (fractured)

    Returns:
        Flat dict with 18 keys (all rounded to 4 decimals).
    """
    # ── Sanitize inputs ────────────────────────────────────
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    if len(y_true) != len(y_pred) or len(y_true) != len(y_prob):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, "
            f"y_prob={len(y_prob)}"
        )
    if len(y_true) == 0:
        raise ValueError("Empty input arrays.")

    # ── Core metrics (macro avg) ───────────────────────────
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ── Per-class metrics ──────────────────────────────────
    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)

    # sklearn returns arrays sorted by class label → index 0 = class 0
    # If a class is missing in y_pred, array may be shorter → pad to length 2
    def _pad(arr, default=0.0):
        a = np.zeros(2, dtype=float)
        a[: len(arr)] = arr[:2]
        return a

    prec_per = _pad(prec_per)
    rec_per = _pad(rec_per)
    f1_per = _pad(f1_per)

    # ── Confusion matrix ───────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm[i, j] = # samples with true=i, pred=j
    # With positive_class=0:
    #   TP = cm[0, 0]  (true fractured, pred fractured)
    #   FN = cm[0, 1]  (true fractured, pred normal)
    #   FP = cm[1, 0]  (true normal,    pred fractured)
    #   TN = cm[1, 1]  (true normal,    pred normal)
    if positive_class == 0:
        tp, fn = int(cm[0, 0]), int(cm[0, 1])
        fp, tn = int(cm[1, 0]), int(cm[1, 1])
    else:
        tp, fn = int(cm[1, 1]), int(cm[1, 0])
        fp, tn = int(cm[0, 1]), int(cm[0, 0])

    # ── Sensitivity / Specificity ─────────────────────────
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ── AUC-ROC ───────────────────────────────────────────
    # sklearn expects y_true in {0,1} with 1 = positive class
    # We transform: (y_true == positive_class) → 1, else → 0
    if len(np.unique(y_true)) < 2:
        auc = 0.0
    else:
        y_true_bin = (y_true == positive_class).astype(int)
        auc = roc_auc_score(y_true_bin, y_prob)

    # ── Return flat dict ──────────────────────────────────
    return {
        # Core (macro averaged)
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec_macro), 4),
        "recall": round(float(rec_macro), 4),
        "f1": round(float(f1_macro), 4),
        # Per-class
        "precision_fractured": round(float(prec_per[0]), 4),
        "precision_normal": round(float(prec_per[1]), 4),
        "recall_fractured": round(float(rec_per[0]), 4),
        "recall_normal": round(float(rec_per[1]), 4),
        "f1_fractured": round(float(f1_per[0]), 4),
        "f1_normal": round(float(f1_per[1]), 4),
        # Threshold-independent
        "auc": round(float(auc), 4),
        # Medical-critical
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        # Confusion matrix
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }