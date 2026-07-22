"""Build Frozen Lake and Forest MDPs used by VI / PI / Q-learning experiments."""
from __future__ import annotations

import random

import matplotlib.pyplot as pt
import numpy as nm
from hiive.mdptoolbox.example import forest

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - legacy fallback
    import gym

random.seed(7)

# Gymnasium renamed FrozenLake8x8-v0 → FrozenLake-v1 with map_name="8x8"
try:
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
except Exception:
    env = gym.make("FrozenLake8x8-v0")

# Prefer unwrapped discrete env for transition table access
_base = env.unwrapped if hasattr(env, "unwrapped") else env
nA = getattr(_base, "nA", None) or _base.action_space.n
nS = getattr(_base, "nS", None) or _base.observation_space.n
_P = _base.P
_desc = _base.desc

PROBS = {"Forest": forest(S=1000, p=0.001, r1=100, r2=10)}

R = nm.zeros([nA, nS, nS])
P = nm.zeros([nA, nS, nS])
DONE = nm.zeros(nS)
for s in range(nS):
    for a in range(nA):
        transitions = _P[s][a]
        for p_trans, next_s, reward, done in transitions:
            P[a, s, next_s] += p_trans
            R[a, s, next_s] += reward
            DONE[next_s] = done
        P[a, s, :] /= nm.sum(P[a, s, :])

PROBS["Frozen_Lake"] = (P, R)

R = nm.zeros([nA, nS, nS])
P = nm.zeros([nA, nS, nS])
DONE = nm.zeros(nS)
for s in range(nS):
    for a in range(nA):
        transitions = _P[s][a]
        for p_trans, next_s, rwrd, done in transitions:
            P[a, s, next_s] += p_trans
            if done and rwrd == 0 and s != next_s:
                rwrd = -0.1  # penalize hole
            R[a, s, next_s] += rwrd
            DONE[next_s] = done
        P[a, s, :] /= nm.sum(P[a, s, :])
PROBS["frozen_lake_modrew"] = (P, R)


def plot_lake(tlist):
    _, ax = pt.subplots(figsize=(5, 5))
    for i in range(8):
        for j in range(8):
            if _desc[i, j] == b"S":
                c = "y"
            elif _desc[i, j] == b"H":
                c = "r"
            elif _desc[i, j] == b"G":
                c = "g"
            elif _desc[i, j] == b"F":
                c = "b"
            p = pt.Rectangle(((j) / 8, (7 - i) / 8), 1 / 8, 1 / 8, color=c)
            ax.add_patch(p)
            t = tlist[i, j]
            try:
                t = t.decode()
            except AttributeError:
                pass
            pt.text(
                (j + 0.5) / 8,
                (7 - i + 0.5) / 8,
                t,
                size=10,
                color="w",
                ha="center",
                va="center",
            )
    pt.xticks([])
    pt.yticks([])
