"""Lightweight correctness checks for MDP transition tensors."""
from __future__ import annotations

import numpy as np


def assert_stochastic(transitions: np.ndarray, atol: float = 1e-8) -> None:
    """Require P[s, a, :] to sum to 1 for every state-action."""
    sums = transitions.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=atol):
        bad = np.argwhere(~np.isclose(sums, 1.0, atol=atol))
        raise AssertionError(f"non-stochastic transitions at indices {bad[:5]}")
