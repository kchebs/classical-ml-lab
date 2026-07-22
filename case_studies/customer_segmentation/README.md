# Case study: Customer segmentation (unsupervised)

Identify population segments that align with a **mail-order** customer base: missing-value handling, scaling, **PCA**, and **K-Means**, then compare general-population vs customer cluster mix.

## Business problem

Mail-order marketing is expensive. Target segments **over-represented among customers** relative to the general population (look-alike audiences).

## Solution

1. Profile missingness via `AZDIAS_Feature_Summary.csv`.
2. Encode / impute / standardize.
3. PCA; choose components by explained variance.
4. K-Means; select `k` via elbow / inertia.
5. Map customers into the same space; highlight over-/under-indexed clusters.

## Data note

Full population/customer extracts are **not redistributable** and are gitignored. For a runnable path:

```bash
python case_studies/customer_segmentation/scripts/bootstrap_data.py
```

Replace generated CSVs with licensed full extracts when available (`AZDIAS_Subset.csv`, `CUSTOMERS_Subset.csv`).

Feature documentation: [`docs/Data_Dictionary.md`](docs/Data_Dictionary.md).

## How to run

```bash
python case_studies/customer_segmentation/scripts/bootstrap_data.py
jupyter notebook case_studies/customer_segmentation/notebooks/customer_segments.ipynb
```
