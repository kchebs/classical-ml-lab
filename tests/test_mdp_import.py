"""Import smoke for sequential_decisions MDP construction (Gymnasium-compatible)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "sequential_decisions"
sys.path.insert(0, str(SD))


def test_make_mdp_builds_frozen_lake_and_forest():
    import make_mdp  # noqa: WPS433 — script-style module

    assert "Forest" in make_mdp.PROBS
    assert "Frozen_Lake" in make_mdp.PROBS
    assert "frozen_lake_modrew" in make_mdp.PROBS
    p, r = make_mdp.PROBS["Frozen_Lake"]
    assert p.ndim == 3 and r.ndim == 3
    assert p.shape == r.shape
