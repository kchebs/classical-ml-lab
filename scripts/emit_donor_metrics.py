#!/usr/bin/env python3
"""Fit a quick logistic model on census smoke data and emit F-beta threshold metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from classicallab.donor_threshold import best_threshold, sweep_fbeta_thresholds

DATA = ROOT / "case_studies" / "donor_prediction" / "data" / "census.csv"
OUT = ROOT / "artifacts" / "donor_threshold_metrics.json"


def main() -> None:
    df = pd.read_csv(DATA)
    y = (df["income"].astype(str).str.contains(">")).astype(int)
    X = df.drop(columns=["income"])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pipe = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), num_cols),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=500, solver="lbfgs")),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    rows = sweep_fbeta_thresholds(y_test.to_numpy(), proba, beta=0.5)
    best = best_threshold(rows)
    payload = {
        "schema_version": 1,
        "beta": 0.5,
        "n_test": int(len(y_test)),
        "best_threshold": best,
        "sweep": rows,
        "note": "Operating point chosen for precision-oriented F0.5 on held-out census fold.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
