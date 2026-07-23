# Case study: Donor prediction (supervised)

Predict whether census respondents earn **more than $50K** to prioritize outreach for a fictional nonprofit. Binary classification on a modified [UCI Adult / Census Income](https://archive.ics.uci.edu/ml/datasets/Census+Income) dataset.

## Business problem

Fundraising budgets are limited. Contact people most likely to donate. Income above $50K correlates with donation likelihood; optimize for **precision-oriented F-beta** to limit wasted outreach.

## Cost-sensitive thresholds

Smoke CI fits a logistic baseline and sweeps probability thresholds, writing [`artifacts/donor_threshold_metrics.json`](../../artifacts/donor_threshold_metrics.json). Alongside F0.5, the artifact reports a simple **cost matrix**: `FP_COST` (wasted outreach dollars) vs `FN_COST` (missed-donor opportunity), mean expected cost per example, the F-beta operating point, and the threshold that minimizes expected cost. Constants live in [`src/classicallab/donor_threshold.py`](../../src/classicallab/donor_threshold.py).

## Feature contract

Column names, dtypes, and roles (numeric / categorical / target) are declared in [`features/donor_features.yaml`](../../features/donor_features.yaml) so preprocessing stays aligned with `data/census.csv`.

## Solution

EDA → log-transform skewed features → scale / one-hot encode → benchmark Naive Bayes → compare Logistic Regression, Random Forest, Gradient Boosting → `GridSearchCV` on F-beta → feature importances.

## How to run

```bash
# from classical-ml-lab root
jupyter notebook case_studies/donor_prediction/notebooks/finding_donors.ipynb
python scripts/emit_donor_metrics.py
```

Plot helpers: [`src/visuals.py`](src/visuals.py). Data: [`data/census.csv`](data/census.csv).
