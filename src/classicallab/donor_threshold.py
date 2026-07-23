"""Cost-sensitive threshold selection for donor outreach (F-beta)."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score


def predict_at_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Binary predictions from positive-class probabilities."""
    return (np.asarray(proba) >= threshold).astype(int)


def sweep_fbeta_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    beta: float = 0.5,
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, float]]:
    """Evaluate precision, recall, and F-beta across probability thresholds."""
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
            }
        )
    return rows


def best_threshold(rows: list[dict[str, float]]) -> dict[str, float]:
    """Return the row with the highest F-beta."""
    if not rows:
        raise ValueError("empty threshold sweep")
    return max(rows, key=lambda r: r["fbeta"])
