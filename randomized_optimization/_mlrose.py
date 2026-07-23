#!/usr/bin/env python3
"""Import mlrose_hiive or exit with an actionable message."""
from __future__ import annotations

import sys


def require_mlrose():
    """Return the mlrose_hiive module, or exit with install guidance."""
    try:
        import mlrose_hiive as mlrose

        return mlrose
    except ImportError as err:
        print(
            "randomized_optimization requires mlrose-hiive with joblib>=1.2,<1.3.\n"
            "  pip install -r requirements-ro.txt\n"
            "If import still fails with 'joblib.my_exceptions', joblib was upgraded\n"
            "(e.g. by scikit-learn); re-run the install above to restore joblib<1.3.\n"
            f"Original error: {err}",
            file=sys.stderr,
        )
        raise SystemExit(2) from err
