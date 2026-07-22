from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "case_studies" / "customer_segmentation"
sys.path.insert(0, str(CASE))

from scripts.bootstrap_data import (  # noqa: E402
    SUMMARY,
    synthesize,
    _parse_missing_codes,
)


def test_smoke_pca_kmeans_pipeline():
    """End-to-end miniature pipeline on schema-compatible synthetic rows."""
    summary = pd.read_csv(SUMMARY, sep=";")
    columns = summary["attribute"].tolist()
    missing_map = {
        row["attribute"]: _parse_missing_codes(str(row["missing_or_unknown"]))
        for _, row in summary.iterrows()
    }
    rng = np.random.default_rng(0)
    az = synthesize(300, columns, missing_map, rng)
    cu = synthesize(80, columns, missing_map, rng)

    for df in (az, cu):
        for col in columns:
            for code in missing_map[col]:
                df[col] = df[col].replace(code, np.nan)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    az_i = imputer.fit_transform(az)
    cu_i = imputer.transform(cu)
    az_s = scaler.fit_transform(az_i)
    cu_s = scaler.transform(cu_i)

    pca = PCA(n_components=10, random_state=0)
    az_p = pca.fit_transform(az_s)
    cu_p = pca.transform(cu_s)
    assert az_p.shape == (300, 10)
    assert cu_p.shape == (80, 10)
    assert pca.explained_variance_ratio_.sum() > 0

    km = KMeans(n_clusters=5, random_state=0, n_init=10)
    az_labels = km.fit_predict(az_p)
    cu_labels = km.predict(cu_p)
    assert set(az_labels).issubset(set(range(5)))
    assert set(cu_labels).issubset(set(range(5)))


def test_bootstrap_writes_expected_files(tmp_path):
    summary = pd.read_csv(SUMMARY, sep=";")
    columns = summary["attribute"].tolist()
    missing_map = {
        row["attribute"]: _parse_missing_codes(str(row["missing_or_unknown"]))
        for _, row in summary.iterrows()
    }
    rng = np.random.default_rng(1)
    out_a = tmp_path / "AZDIAS_Subset.csv"
    out_c = tmp_path / "CUSTOMERS_Subset.csv"
    synthesize(50, columns, missing_map, rng).to_csv(out_a, sep=";", index=False)
    synthesize(20, columns, missing_map, rng).to_csv(out_c, sep=";", index=False)
    assert pd.read_csv(out_a, sep=";").shape[1] == len(columns)
    assert pd.read_csv(out_c, sep=";").shape[1] == len(columns)
