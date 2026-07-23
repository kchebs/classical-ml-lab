#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Prefer active venv, then local .venv, then python3 on PATH
# Prefer this repo's .venv over an ambient VIRTUAL_ENV from another project
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PY="${VIRTUAL_ENV}/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

# On Apple Silicon, parent shells under Rosetta can pick the x86_64 slice of a
# universal Python while wheels are arm64 — force arm64 when available.
run_py() {
  if command -v arch >/dev/null 2>&1 && arch -arm64 /usr/bin/true >/dev/null 2>&1; then
    arch -arm64 "$PY" "$@"
  else
    "$PY" "$@"
  fi
}

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
echo "== classical-ml-lab smoke (interpreter: $PY) =="

run_py - <<'PY'
from pathlib import Path
root = Path(".")
required = [
    "supervised_learning/data/breastcancer.csv",
    "supervised_learning/data/titanic.csv",
    "supervised_learning/notebooks/supervised_comparison.ipynb",
    "supervised_learning/docs/analysis.md",
    "randomized_optimization/ff.py",
    "randomized_optimization/ks.py",
    "randomized_optimization/nn.py",
    "randomized_optimization/ts.py",
    "randomized_optimization/docs/analysis.md",
    "unsupervised_learning/data/breastcancer.csv",
    "unsupervised_learning/notebooks/unsupervised_dimensionality.ipynb",
    "unsupervised_learning/docs/analysis.md",
    "sequential_decisions/run.py",
    "sequential_decisions/docs/analysis.md",
    "case_studies/donor_prediction/data/census.csv",
    "case_studies/donor_prediction/notebooks/finding_donors.ipynb",
    "case_studies/donor_prediction/src/visuals.py",
    "case_studies/customer_segmentation/data/AZDIAS_Feature_Summary.csv",
    "case_studies/customer_segmentation/notebooks/customer_segments.ipynb",
    "case_studies/customer_segmentation/scripts/bootstrap_data.py",
]
missing = [p for p in required if not (root / p).is_file()]
if missing:
    raise SystemExit("Missing required files:\n" + "\n".join(missing))
print(f"OK layout ({len(required)} required paths present)")
PY

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
echo "== bootstrap customer smoke data =="
run_py case_studies/customer_segmentation/scripts/bootstrap_data.py

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
echo "== pytest =="
run_py -m pytest -q

run_py scripts/emit_donor_metrics.py
echo "SMOKE PASS"
