"""Lightweight structural checks for classical methods modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_supervised_data_and_notebook_present():
    assert (ROOT / "supervised_learning" / "data" / "breastcancer.csv").is_file()
    assert (ROOT / "supervised_learning" / "data" / "titanic.csv").is_file()
    assert (
        ROOT / "supervised_learning" / "notebooks" / "supervised_comparison.ipynb"
    ).is_file()
    assert (ROOT / "supervised_learning" / "docs" / "analysis.md").is_file()


def test_randomized_optimization_scripts_present():
    for name in ("ff.py", "ks.py", "nn.py", "ts.py"):
        assert (ROOT / "randomized_optimization" / name).is_file()
    assert (ROOT / "randomized_optimization" / "data" / "breastcancer.csv").is_file()
    assert (ROOT / "randomized_optimization" / "docs" / "analysis.md").is_file()


def test_unsupervised_data_and_notebook_present():
    assert (ROOT / "unsupervised_learning" / "data" / "breastcancer.csv").is_file()
    assert (
        ROOT / "unsupervised_learning" / "notebooks" / "unsupervised_dimensionality.ipynb"
    ).is_file()
    assert (ROOT / "unsupervised_learning" / "docs" / "analysis.md").is_file()


def test_sequential_decisions_scripts_present():
    for name in ("run.py", "make_mdp.py", "val_iter.py", "policy_iter.py", "q_fc.py"):
        assert (ROOT / "sequential_decisions" / name).is_file()
    assert (ROOT / "sequential_decisions" / "docs" / "analysis.md").is_file()
