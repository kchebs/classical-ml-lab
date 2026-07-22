# Migration notes

This repository consolidates three prior folders into one MECE classical-ML lab:

| Former location | New location |
|-----------------|--------------|
| Classical supervised methods folder | `supervised_learning/` |
| Classical randomized-optimization folder | `randomized_optimization/` |
| Classical unsupervised / DR folder | `unsupervised_learning/` |
| Classical MDP / Q-learning folder | `sequential_decisions/` |
| Standalone donor-prediction repo | `case_studies/donor_prediction/` |
| Standalone customer-segmentation repo | `case_studies/customer_segmentation/` |

## Parity preserved

- Supervised comparison notebook + Breast Cancer / Titanic data + analysis notes (Markdown)
- RO scripts (FlipFlop, Knapsack, TSP, NN weights) + result/asset artifacts
- Unsupervised notebook + data + cluster center exports + analysis notes
- MDP / Q-learning scripts + committed scorecard assets + analysis notes
- Donor census data, notebook, `visuals.py`, schema tests
- Segmentation feature summary, bootstrap smoke synthesizer, notebook, dictionary, PCA→K-Means tests

## Improvements vs the triad

- Single root README and thematic modules (production-style layout)
- Word/PDF reports converted to Markdown with personal identifiers removed
- Gymnasium-compatible Frozen Lake MDP construction
- Repo-wide `scripts/smoke.sh` + pytest covering layout, cases, and MDP import
- `mlrose_hiive` import shims and path-stable RO scripts
- Former standalone donor / segmentation repos removed after merge
