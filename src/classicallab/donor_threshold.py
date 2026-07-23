"""Cost-sensitive threshold selection for donor outreach (F-beta + dollar costs)."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score

# Dollar weights for a fictional nonprofit outreach campaign.
# FP: wasted mailer / staff time contacting a non-donor.
# FN: foregone expected gift from a missed high-income prospect.
# FP > FN encodes a precision-first operating preference (limit wasted outreach).
FP_COST: float = 25.0
FN_COST: float = 10.0


def predict_at_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Binary predictions from positive-class probabilities."""
    return (np.asarray(proba) >= threshold).astype(int)


def expected_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> float:
    """Mean dollar cost per example: (FP * fp_cost + FN * fn_cost) / n."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    n = int(y_true.size)
    if n == 0:
        return 0.0
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return float((fp * fp_cost + fn * fn_cost) / n)


def sweep_fbeta_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    beta: float = 0.5,
    thresholds: Iterable[float] | None = None,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> list[dict[str, float]]:
    """Evaluate precision, recall, F-beta, and expected cost across thresholds."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    rows: list[dict[str, float]] = []
    for t in thresholds:
        y_hat = predict_at_threshold(proba, float(t))
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_hat, zero_division=0)),
                "recall": float(recall_score(y_true, y_hat, zero_division=0)),
                "fbeta": float(fbeta_score(y_true, y_hat, beta=beta, zero_division=0)),
                "expected_cost": expected_cost(y_true, y_hat, fp_cost, fn_cost),
            }
        )
    return rows


def best_threshold(rows: list[dict[str, float]]) -> dict[str, float]:
    """Return the row with the highest F-beta."""
    if not rows:
        raise ValueError("empty threshold sweep")
    return max(rows, key=lambda r: r["fbeta"])


def best_threshold_by_cost(rows: list[dict[str, float]]) -> dict[str, float]:
    """Return the row with the lowest expected cost."""
    if not rows:
        raise ValueError("empty threshold sweep")
    return min(rows, key=lambda r: r["expected_cost"])
