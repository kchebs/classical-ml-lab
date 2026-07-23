"""Algorithm test: Frozen Lake transitions are stochastic."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sequential_decisions"))
sys.path.insert(0, str(ROOT / "src"))

from classicallab.mdp_checks import assert_stochastic


def test_frozen_lake_transitions_stochastic():
    import make_mdp

    p, _r = make_mdp.PROBS["Frozen_Lake"]
    assert_stochastic(p)
