# Randomized Optimization

Discrete optimization with randomized hill climbing, simulated annealing, genetic algorithms, and MIMIC on FlipFlop, Knapsack, and TSP — plus neural-network weight search compared to gradient descent.

## Business framing

Many product and ops problems are combinatorial (assortment, routing, configuration). This module studies **when randomized search beats local greedy methods** and how NN weight optimization via RO compares to gradient-based training.

## Dependencies

Core lab install is enough to browse assets and docs. To **execute** the RO runners:

```bash
# Prefer Python 3.10–3.12; mlrose-hiive is fragile on 3.13+
pip install -r requirements-ro.txt
```

## How to run

```bash
# from classical-ml-lab root, with venv active
python randomized_optimization/ff.py
python randomized_optimization/ks.py
python randomized_optimization/ts.py
python randomized_optimization/nn.py
```

Scripts resolve `data/`, `results/`, and `assets/` relative to this folder. Full runs can be slow.

Analysis notes: [`docs/analysis.md`](docs/analysis.md).
