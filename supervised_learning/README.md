# Supervised Learning

Compare SVM, neural nets, kNN, decision trees, and boosted trees on two binary classification datasets: Breast Cancer and Titanic.

## Business framing

Medical risk triage and survival prediction are different domains with different feature strengths. Running the **same learners** on both surfaces algorithm bias/variance and dataset difficulty—not a single leaderboard number.

## How to run

```bash
# from classical-ml-lab root, with venv active
jupyter notebook supervised_learning/notebooks/supervised_comparison.ipynb
```

Data: [`data/breastcancer.csv`](data/breastcancer.csv), [`data/titanic.csv`](data/titanic.csv).

Analysis notes: [`docs/analysis.md`](docs/analysis.md).
