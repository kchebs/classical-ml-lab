import pandas as da
from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration
import matplotlib.pyplot as pt
from make_mdp import PROBS
from q_fc import *

for prob_ky in PROBS:
    if prob_ky not in ['Forest','Frozen_Lake']:
        continue
    print(f"Running {prob_ky}...")
    P, R = PROBS[prob_ky]

    if prob_ky == "Frozen_Lake":
        n_epi = 10000
        eps_schedule = make_schedules(n_epi)["exp_decay"]
        alpha_schedule = make_schedules(n_epi)["constant_0.01"]
    elif prob_ky == "Forest":
        n_epi = 100000
        eps_schedule = make_schedules(n_epi)["constant_0.5"]
        alpha_schedule = make_schedules(n_epi)["constant_0.5"]

    print("Value Iteration...")
    vi = ValueIteration(P, R, gamma=0.99, epsilon=0.001, max_iter=1000)
    vi.run()
    vi_df = da.DataFrame(vi.run_stats).set_index("Iteration")
    vi_df.columns = da.MultiIndex.from_product([vi_df.columns, ["value_iter"]])
    print(f"Runtime per value iter: {vi.time/vi.iter} sec")

    print("Policy Iteration...")
    pi = PolicyIteration(P, R, gamma=0.99, eval_type=1, max_iter=1000)
    pi.run()
    pi_df = da.DataFrame(pi.run_stats).set_index("Iteration")
    pi_df.columns = da.MultiIndex.from_product([pi_df.columns, ["policy_iter"]])
    print(f"Runtime per policy iter: {pi.time/pi.iter} sec")

    print("Q-learning...")
    ql = QLearning(P, R, gamma=0.99, alpha_schedule=alpha_schedule, epsilon_schedule=eps_schedule, n_episode=n_epi)
    random.seed(7)
    ql.run()
    ql_df = da.DataFrame(ql.run_stats)
    ql_df["Alpha"] = alpha_schedule
    ql_df["Epsilon"] = eps_schedule
    ql_df[["Reward_rolling_mean", "Error_rolling_mean"]] = (
        ql_df[["Reward", "Error"]].rolling(1000).mean()
    )
    ql_df[
        [
            "Reward_rolling_mean",
            "Error_rolling_mean",
            "Time",
            "Alpha",
            "Epsilon",
            "Max V",
            "Mean V",
        ]
    ].plot(subplots=True, title=f"{prob_ky} Q-Learning Training")
    pt.xlabel("Episode")
    pt.savefig(f"{prob_ky}_q.png")
    pt.close()
    print(f"Runtime per q-learning episode: {ql.time/ql.n_episode} sec")

    res_df = vi_df.join(pi_df, how="outer")
    for var in ["Error", "Time", "Max V", "Mean V"]:
        res_df.xs(var, level=0, axis=1).plot(
            title=f"{var} on {prob_ky}", figsize=(8, 4)
        )
        pt.savefig(f"{prob_ky}_{var.replace(' ','')}.png")
        pt.close()

    print(f"Policy diff vi vs. pi: {sum([p1!=p2 for p1,p2 in zip(vi.policy, pi.policy)])}")
    print(f"Policy diff vi vs. ql: {sum([p1!=p2 for p1,p2 in zip(vi.policy, ql.policy)])}")
    print(f"Policy diff pi vs. ql: {sum([p1!=p2 for p1,p2 in zip(pi.policy, ql.policy)])}")
    print(f"Value diff vi vs. pi: {sum([abs(v1-v2) for v1,v2 in zip(vi.V, pi.V)])}")
    print(f"Value diff vi vs. ql: {sum([abs(v1-v2) for v1,v2 in zip(vi.V, ql.V)])}")
    print(f"Value diff pi vs. ql: {sum([abs(v1-v2) for v1,v2 in zip(pi.V, ql.V)])}")

    V_df = da.DataFrame(data=[vi.V, pi.V, ql.V], index=["vi", "pi", "ql"]).T
    V_df.to_csv(f"{prob_ky}_V_cmp.csv")

    policy_df = da.DataFrame(
        data=[vi.policy, pi.policy, ql.policy], index=["vi", "pi", "ql"]
    ).T
    policy_df.to_csv(f"{prob_ky}_policy_cmp.csv")