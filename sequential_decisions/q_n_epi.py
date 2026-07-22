import pandas as da
import matplotlib.pyplot as pt
from make_mdp import PROBS
from q_fc import *

random.seed(7)
for prob_ky in PROBS:
    if prob_ky not in ['Forest','Frozen_Lake']:
        continue
    P, R = PROBS[prob_ky]

    if prob_ky == "Frozen_Lake":
        gamma = 0.99
        N_EPISODES = [100, 1000, 5000, 10000, 100000]

    elif prob_ky == "Forest":
        gamma = 0.99
        N_EPISODES = [100, 1000, 10000, 50000, 100000, 1000000]

    res_dict = {
        "Mean Reward": [],
        "Mean dQ": [],
        "Max V": [],
        "Mean V": [],
        "Optimal policy reward": [],
    }
    policy_all = []; V_all = []; r_std_all = []

    for n_epi in N_EPISODES:
        # constant learning rate
        alpha_schedule = [0.01] * n_epi
        # all exploration
        epsilon_schedule = [0.5] * n_epi
        ql = QLearning(P, R, gamma=gamma, n_episode=n_epi, alpha_schedule=alpha_schedule, epsilon_schedule=epsilon_schedule)
        random.seed(0)
        ql.run()
        res_dict["Mean Reward"].append(nm.mean(ql.run_stats["Reward"][-1000:]))
        res_dict["Mean dQ"].append(nm.mean(ql.run_stats["Error"][-1000:]))
        res_dict["Max V"].append(ql.run_stats["Max V"][-1])
        res_dict["Mean V"].append(ql.run_stats["Mean V"][-1])
        test_r_mean, test_r_std = test_policy(P, R, ql.policy)
        res_dict["Optimal policy reward"].append(test_r_mean)
        r_std_all.append(test_r_std)
        V_all.append(ql.V)
        policy_all.append(ql.policy)
    resdf = da.DataFrame(res_dict)
    resdf.to_csv(f'{prob_ky}_q_resdf.csv')
    resdf.index = nm.array(N_EPISODES).astype(str)
    resdf.plot(
        subplots=True,
        title=f"Q-Learning vs. training episodes on {prob_ky}",
        style=".-",
        figsize=(7, 7))
    pt.xlabel("Number of Episodes")
    pt.savefig(f"{prob_ky}_ql_n_epi.png")
    pt.close()