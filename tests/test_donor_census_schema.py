from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "case_studies" / "donor_prediction" / "data" / "census.csv"


def test_census_file_exists():
    assert DATA.is_file(), f"Missing dataset at {DATA}"


def test_census_has_expected_columns_and_target():
    df = pd.read_csv(DATA)
    assert "income" in df.columns
    assert len(df) > 1000
    assert df["income"].nunique() >= 2
