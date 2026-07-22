#!/usr/bin/env python3
"""
Create local population / customer CSVs for a runnable smoke path.

Full demographics extracts used by the notebook are not redistributable and are
not hosted on GitHub. This script synthesizes schema-compatible CSVs from
data/AZDIAS_Feature_Summary.csv so cleaning → PCA → K-Means can be demonstrated.

Replace the generated files with licensed full extracts under data/ when available
(filenames: AZDIAS_Subset.csv, CUSTOMERS_Subset.csv).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data" / "AZDIAS_Feature_Summary.csv"
AZDIAS_OUT = ROOT / "data" / "AZDIAS_Subset.csv"
CUSTOMERS_OUT = ROOT / "data" / "CUSTOMERS_Subset.csv"


def _parse_missing_codes(raw: str) -> list[int]:
    raw = (raw or "").strip()
    if not raw or raw == "[]":
        return []
    inner = raw.strip("[]")
    codes: list[int] = []
    for part in inner.split(","):
        part = part.strip()
        if not part or part == "X" or part == "XX":
            continue
        try:
            codes.append(int(part))
        except ValueError:
            continue
    return codes


def synthesize(n_rows: int, columns: list[str], missing_map: dict[str, list[int]], rng: np.random.Generator) -> pd.DataFrame:
    data = {}
    for col in columns:
        codes = missing_map.get(col, [])
        # Prefer small positive ints like census ordinals/categoricals
        values = rng.integers(1, 6, size=n_rows)
        if codes:
            # Sprinkle coded missings (~8%)
            mask = rng.random(n_rows) < 0.08
            values = values.astype(object)
            values[mask] = codes[0]
        data[col] = values
    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--azdias-rows", type=int, default=800)
    parser.add_argument("--customers-rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing CSVs even if present",
    )
    args = parser.parse_args()

    if not SUMMARY.is_file():
        raise SystemExit(f"Missing feature summary: {SUMMARY}")

    summary = pd.read_csv(SUMMARY, sep=";")
    columns = summary["attribute"].tolist()
    missing_map = {
        row["attribute"]: _parse_missing_codes(str(row["missing_or_unknown"]))
        for _, row in summary.iterrows()
    }
    rng = np.random.default_rng(args.seed)

    for path, n in ((AZDIAS_OUT, args.azdias_rows), (CUSTOMERS_OUT, args.customers_rows)):
        if path.is_file() and not args.force:
            print(f"Keeping existing {path} (pass --force to overwrite)")
            continue
        df = synthesize(n, columns, missing_map, rng)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, sep=";", index=False)
        print(f"Wrote {path} shape={df.shape}")

    # Sanity: notebook load paths
    az = pd.read_csv(AZDIAS_OUT, sep=";")
    cu = pd.read_csv(CUSTOMERS_OUT, sep=";")
    assert list(az.columns) == columns
    assert list(cu.columns) == columns
    print(f"OK smoke data ready: azdias={az.shape} customers={cu.shape}")


if __name__ == "__main__":
    main()
