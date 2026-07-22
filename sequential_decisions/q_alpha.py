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
        n_epi = 10000
        eps_schedule = make_schedules(n_epi)["exp_decay"]
        alpha_schedules = make_schedules(n_epi)

    elif prob_ky == "Forest":
        gamma = 0.99
        n_epi = 100000
        eps_schedule = make_schedules(n_epi)["constant_0.5"]
        alpha_schedules = make_schedules(n_epi)

    res_dict = {
        "Mean Reward": [],
        "Mean dQ": [],
        "Max V": [],
        "Mean V": [],
        "Optimal policy reward": [],
        "alpha schedule key": [],
    }
    policy_all = []; V_all = []; r_std_all = []
    for ky in alpha_schedules:
        ql = QLearning(P, R, gamma=gamma, n_episode=n_epi, alpha_schedule=alpha_schedules[ky], epsilon_schedule=eps_schedule)
        random.seed(7)
        ql.run()
        res_dict["Mean Reward"].append(nm.mean(ql.run_stats["Reward"][-1000:]))
        res_dict["Mean dQ"].append(nm.mean(ql.run_stats["Error"][-1000:]))
        res_dict["Max V"].append(ql.run_stats["Max V"][-1])
        res_dict["Mean V"].append(ql.run_stats["Mean V"][-1])
        res_dict["alpha schedule key"].append(ky)
        test_r_mean, test_r_std = test_policy(P, R, ql.policy)
        res_dict["Optimal policy reward"].append(test_r_mean)
        r_std_all.append(test_r_std)
        V_all.append(ql.V)
        policy_all.append(ql.policy)
    res_df = da.DataFrame(res_dict).set_index("alpha schedule key")
    res_df.plot(subplots=True, title=f"{prob_ky} Q-Learning vs. alpha schedule", style=".-", figsize=(7, 7))
    pt.xlabel("Number of Episodes")
    pt.savefig(f"{prob_ky}_ql_alpha.png")
    pt.close()

    da.DataFrame(alpha_schedules).plot(title=f"{prob_ky} Q-Learning Alpha Schedule")
    pt.xlabel("Number of Episodes")
    pt.ylabel("alpha")
    pt.savefig(f"{prob_ky}_ql_alpha_schedule.png")
    pt.close()
