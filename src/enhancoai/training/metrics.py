"""Classification/regression metrics reported by the training GUI (section 41)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, mean_absolute_error, r2_score


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {}
    try:
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except ValueError:
        metrics["f1"] = float("nan")
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        metrics["auroc"] = float("nan")
    try:
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        metrics["mcc"] = float("nan")
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {"mae": float(mean_absolute_error(y_true, y_pred))}
    try:
        metrics["r2"] = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    except ValueError:
        metrics["r2"] = float("nan")
    return metrics
