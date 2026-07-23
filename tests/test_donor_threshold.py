"""Algorithm tests for cost-sensitive donor thresholds."""
from __future__ import annotations

import numpy as np

from classicallab.donor_threshold import (
    FN_COST,
    FP_COST,
    best_threshold,
    best_threshold_by_cost,
    expected_cost,
    predict_at_threshold,
    sweep_fbeta_thresholds,
)


def test_predict_at_threshold():
    proba = np.array([0.1, 0.4, 0.6, 0.9])
    assert list(predict_at_threshold(proba, 0.5)) == [0, 0, 1, 1]


def test_sweep_prefers_high_precision_when_separable():
    # Perfectly separable scores — high threshold remains high F0.5
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
    rows = sweep_fbeta_thresholds(y, p, beta=0.5, thresholds=[0.3, 0.5, 0.7])
    best = best_threshold(rows)
    assert best["precision"] == 1.0
    assert best["fbeta"] > 0.9
    assert "expected_cost" in best


def test_expected_cost_fp_and_fn():
    y_true = np.array([0, 0, 1, 1])
    # one FP, one FN
    y_pred = np.array([0, 1, 0, 1])
    cost = expected_cost(y_true, y_pred, fp_cost=10.0, fn_cost=20.0)
    assert cost == (10.0 + 20.0) / 4.0


def test_best_threshold_by_cost_avoids_expensive_fps():
    y = np.array([0, 0, 0, 0, 1, 1])
    # Mid scores are negatives: low threshold incurs FPs; high threshold is cheaper
    p = np.array([0.2, 0.25, 0.55, 0.6, 0.8, 0.9])
    rows = sweep_fbeta_thresholds(
        y, p, beta=0.5, thresholds=[0.15, 0.5, 0.75], fp_cost=100.0, fn_cost=1.0
    )
    best_cost = best_threshold_by_cost(rows)
    assert best_cost["threshold"] == 0.75
    assert best_cost["expected_cost"] < rows[0]["expected_cost"]
    assert FP_COST > FN_COST
