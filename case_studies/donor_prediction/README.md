# Case study: Donor prediction (supervised)

Predict whether census respondents earn **more than $50K** to prioritize outreach for a fictional nonprofit. Binary classification on a modified [UCI Adult / Census Income](https://archive.ics.uci.edu/ml/datasets/Census+Income) dataset.

## Business problem

Fundraising budgets are limited. Contact people most likely to donate. Income above $50K correlates with donation likelihood; optimize for **precision-oriented F-beta** to limit wasted outreach.

## Solution

EDA → log-transform skewed features → scale / one-hot encode → benchmark Naive Bayes → compare Logistic Regression, Random Forest, Gradient Boosting → `GridSearchCV` on F-beta → feature importances.

## How to run

```bash
# from classical-ml-lab root
jupyter notebook case_studies/donor_prediction/notebooks/finding_donors.ipynb
```

Plot helpers: [`src/visuals.py`](src/visuals.py). Data: [`data/census.csv`](data/census.csv).
