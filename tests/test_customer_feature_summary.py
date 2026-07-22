from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "case_studies"
    / "customer_segmentation"
    / "data"
    / "AZDIAS_Feature_Summary.csv"
)


def test_feature_summary_exists_and_has_rows():
    assert SUMMARY.is_file()
    df = pd.read_csv(SUMMARY, sep=";")
    assert len(df) >= 50
