import numpy as nm
import pandas as da
from hiive.mdptoolbox.mdp import PolicyIteration
import matplotlib.pyplot as pt
from make_mdp import PROBS
import random
from q_fc import test_policy


disc_rt = [0.1, 0.25, 0.9, 0.99]
random.seed(7)
for prob_ky in PROBS:
    if prob_ky not in ['Forest','Frozen_Lake']:
        continue
    P, R = PROBS[prob_ky]

    res_dict = {
        "Iteration to converge": [],
        "Max V": [],
        "Mean V": [],
        "Optimal policy reward": [],
    }
    V_all = []; policy_all = []; r_std_all = []
    for g in disc_rt:
        pi = PolicyIteration(P, R, gamma=g, eval_type=1, max_iter=1000)
        pi.run()
        res_dict["Iteration to converge"].append(pi.iter)
        res_dict["Max V"].append(pi.run_stats[-1]["Max V"])
        res_dict["Mean V"].append(pi.run_stats[-1]["Mean V"])
        test_r_mean, test_r_std = test_policy(P, R, pi.policy)
        res_dict["Optimal policy reward"].append(test_r_mean)
        r_std_all.append(test_r_std)
        V_all.append(pi.V)
        policy_all.append(pi.policy)
    res_df = da.DataFrame(res_dict)
    res_df.to_csv(f'{prob_ky}_policy_resdf.csv')
    res_df.index = nm.array(disc_rt).astype(str)
    res_df.plot(
        subplots=True,
        title=f"{prob_ky} Policy Iteration",
        style=".-",
        figsize=(7, 7),
    )
    pt.xlabel("Discount Rate")
    pt.savefig(f"{prob_ky}_pi_gamma.png")
    pt.close()

    da.DataFrame(V_all, index=res_df.index).T.to_csv(f"{prob_ky}_pi_V.csv")
    da.DataFrame(policy_all, index=res_df.index).T.to_csv(f"{prob_ky}_pi_policy.csv")