import numpy as nm
import pandas as da
from hiive.mdptoolbox.mdp import ValueIteration
import matplotlib.pyplot as pt
from make_mdp import PROBS
import random
from q_fc import test_policy

disc_rt = [0.1, 0.25, 0.9, 0.99]
random.seed(7)
for prob_ky in PROBS:
    if prob_ky not in ['Forest','Frozen_Lake']:
        continue
    print(f"Running {prob_ky}...")
    P, R = PROBS[prob_ky]

    print("..Running value iteration...")
    res_dict = {
        "Iteration to converge": [],
        "Max V": [],
        "Mean V": [],
        "Optimal policy reward": [],
    }
    V_all = []; policy_all = []; r_std_all = []
    for g in disc_rt:
        print(f"..discount rate = {g}...")
        vi = ValueIteration(P, R, gamma=g, epsilon=0.001)
        vi.run()
        res_dict["Iteration to converge"].append(vi.iter)
        res_dict["Max V"].append(vi.run_stats[-1]["Max V"])
        res_dict["Mean V"].append(vi.run_stats[-1]["Mean V"])
        test_r_mean, test_r_std = test_policy(P, R, vi.policy)
        res_dict["Optimal policy reward"].append(test_r_mean)
        r_std_all.append(test_r_std)
        V_all.append(vi.V)
        policy_all.append(vi.policy)
    res_df = da.DataFrame(res_dict)
    res_df.to_csv(f'{prob_ky}_value_resdf.csv')
    res_df.index = nm.array(disc_rt).astype(str)
    res_df.plot(
        subplots=True,
        title=f"{prob_ky} Value iteration",
        style=".-",
        figsize=(7, 7),
    )
    pt.xlabel("Discount Rate")
    pt.savefig(f"{prob_ky}_vi_gamma.png")
    pt.close()

    da.DataFrame(policy_all, index=res_df.index).T.to_csv(f"{prob_ky}_vi_policy.csv")
    da.DataFrame(V_all, index=res_df.index).T.to_csv(f"{prob_ky}_vi_V.csv")
