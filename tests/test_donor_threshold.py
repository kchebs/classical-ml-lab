"""Algorithm tests for cost-sensitive donor thresholds."""
from __future__ import annotations

import numpy as np

from classicallab.donor_threshold import best_threshold, predict_at_threshold, sweep_fbeta_thresholds


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
