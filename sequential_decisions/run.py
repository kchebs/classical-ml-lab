from pathlib import Path
import sys
_SD_ROOT = Path(__file__).resolve().parent
if str(_SD_ROOT) not in sys.path:
    sys.path.insert(0, str(_SD_ROOT))
import val_iter
import numpy as nm
import policy_iter
import q_n_epi
import pandas as da
import q_epsilon
import matplotlib.pyplot as pt
import q_alpha
from make_mdp import plot_lake
import compare
import n_states
import random

random.seed(7)
policy = da.read_csv("Frozen_Lake_vi_policy.csv")["0.99"]
actions = nm.array(["<v>^"[i] for i in policy])
plot_lake(actions.reshape(8,8))
pt.title("Frozen Lake Value Iteration Map")
pt.savefig("frzn_lake_val_map.png")
pt.close()

policy = da.read_csv("Frozen_Lake_pi_policy.csv")["0.99"]
actions = nm.array(["<v>^"[i] for i in policy])
plot_lake(actions.reshape(8,8))
pt.title("Frozen Lake Policy Map")
pt.savefig("frzn_lake_policy_map.png")
pt.close()

policy = da.read_csv("Frozen_Lake_policy_cmp.csv")["ql"]
actions = nm.array(["<v>^"[i] for i in policy])
plot_lake(actions.reshape(8,8))
pt.title("Q-Learning Frozen Lake Policy Map")
pt.savefig("frzn_lake_q_policy_map.png")
pt.close()

for prob_ky in ["Frozen_Lake", "Forest"]:
    vi_V = da.read_csv(f"{prob_ky}_vi_V.csv", index_col=0)
    pi_V = da.read_csv(f"{prob_ky}_pi_V.csv", index_col=0)
    _, axes = pt.subplots(vi_V.shape[1], 1, sharex=True, figsize=(8, 8), gridspec_kw={"hspace": 0.7})
    i = 0
    for cl in vi_V.columns:
        ax = axes[i]
        ax.plot(vi_V[cl])
        ax.plot(pi_V[cl], ":")
        ax.set_title(f"gamma={cl}")
        i += 1
    pt.legend(["Value Iteration", "Policy Iteration"])
    pt.xlabel("State")
    pt.suptitle("Value Function")
    pt.savefig(f"{prob_ky}_V_vi_pi.png")
    pt.close()

    V_diff = vi_V - pi_V
    axes = V_diff.plot(subplots=True,
        title="Value function difference between two iteration types", figsize=(7, 7))
    for ax in axes:
        ax.ticklabel_format(useOffset=False)
    pt.xlabel("State")
    pt.savefig(f"{prob_ky}_V_diff.png")
    pt.close()

    vi_policy = da.read_csv(f"{prob_ky}_vi_policy.csv", index_col=0)
    pi_policy = da.read_csv(f"{prob_ky}_pi_policy.csv", index_col=0)
    _, axes = pt.subplots(vi_policy.shape[1], 1, sharex=True, figsize=(7, 7), gridspec_kw={"hspace": 0.7})
    i = 0
    for cl in vi_V.columns:
        ax = axes[i]
        ax.plot(vi_policy[cl], "-")
        ax.plot(pi_policy[cl], ":")
        ax.set_title(f"gamma={cl}")
        i += 1
    pt.legend(["Value Iteration", "Policy Iteration"])
    pt.xlabel("State")
    pt.suptitle("Policy from value iteration and policy iteration")
    pt.savefig(f"{prob_ky}_policy_vi_pi.png")
    pt.close()


n_states_vi = da.read_csv("forest_val_nS.csv", index_col=0)
n_states_pi = da.read_csv("forest_pol_nS.csv", index_col=0)

n_states_vi["Time per Iteration"] = (n_states_vi["Time to converge"] / n_states_vi["Iteration to converge"])
n_states_vi.plot(subplots=True, style=".-", title="Value Iteration")
pt.xlabel("Number of States")
pt.savefig("val_nS.png")
pt.close()

n_states_pi["Time per Iteration"] = (n_states_pi["Time to converge"] / n_states_pi["Iteration to converge"])
n_states_pi.plot(subplots=True, style=".-", title="Policy Iteration")
pt.xlabel("Number of States")
pt.savefig("pol_nS.png")
pt.close()
