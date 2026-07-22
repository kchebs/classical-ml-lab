import pandas as da
import matplotlib.pyplot as pt
from make_mdp import PROBS
from q_fc import *

random.seed(7)
for prob_ky in PROBS:
    if prob_ky not in ["Frozen_lake"]:
        continue
    P, R = PROBS[prob_ky]

    if prob_ky == "Frozen_Lake":
        gamma = 0.99
        n_epi = 10000
        epsilon_schedules = make_schedules(n_epi)

    elif prob_ky == "Forest":
        gamma = 0.99
        n_epi = 100000
        epsilon_schedules = make_schedules(n_epi)

    res_dict = {
        "Mean Reward": [],
        "Mean dQ": [],
        "Max V": [],
        "Mean V": [],
        "Optimal policy reward": [],
        "eps schedule key": [],
    }
    V_all = []; policy_all = []; r_std_all = []

    for ky in epsilon_schedules:
        alpha_schedule = [0.01] * n_epi
        eps_schedule = epsilon_schedules[ky]
        ql = QLearning(P, R, gamma=gamma, n_episode=n_epi, alpha_schedule=alpha_schedule, epsilon_schedule=eps_schedule)
        random.seed(0)
        ql.run()
        res_dict["Mean Reward"].append(nm.mean(ql.run_stats["Reward"][-1000:]))
        res_dict["Mean dQ"].append(nm.mean(ql.run_stats["Error"][-1000:]))
        res_dict["Max V"].append(ql.run_stats["Max V"][-1])
        res_dict["Mean V"].append(ql.run_stats["Mean V"][-1])
        res_dict["eps schedule key"].append(ky)
        test_r_mean, test_r_std = test_policy(P, R, ql.policy)
        res_dict["Optimal policy reward"].append(test_r_mean)
        r_std_all.append(test_r_std)
        V_all.append(ql.V)
        policy_all.append(ql.policy)
    res_df = da.DataFrame(res_dict).set_index("eps schedule key")
    res_df.plot(subplots=True, title=f"{prob_ky} Q-Learning vs Exploration Schedule", style=".-", figsize=(7, 7))
    pt.xlabel("Number of Episodes")
    pt.savefig(f"{prob_ky}_ql_eps.png")
    pt.close()

    da.DataFrame(epsilon_schedules).plot(title=f"{prob_ky} Q-Learning Epsilon Schedule")
    pt.xlabel("Number of Episodes")
    pt.ylabel("Epsilon")
    pt.savefig(f"{prob_ky}_ql_eps_schedule.png")
    pt.close()
