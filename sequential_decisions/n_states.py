import pandas as da
from hiive.mdptoolbox.example import forest
import random
from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration

policy_all = {}; V_all = {}
vi_res = {
    "Iteration to converge": [],
    "Time to converge": [],
    "Max V": [],
    "Mean V": [],
}
num_states = range(100, 1001, 100)
pi_res = {
    "Iteration to converge": [],
    "Time to converge": [],
    "Max V": [],
    "Mean V": [],
}
random.seed(7)
for s in num_states:
    print(f"Running nS {s}...")
    P, R = forest(S=s, p=0.001, r1=100, r2=10)
    vi = ValueIteration(P, R, gamma=0.99, epsilon=0.001)
    vi.run()
    vi_res["Iteration to converge"].append(vi.iter)
    vi_res["Time to converge"].append(vi.time)
    vi_res["Max V"].append(vi.run_stats[-1]["Max V"])
    vi_res["Mean V"].append(vi.run_stats[-1]["Mean V"])
    V_all[("vi", s)] = vi.V
    policy_all[("vi", s)] = vi.policy
    pi = PolicyIteration(P, R, gamma=0.99, eval_type=1, max_iter=1000)
    pi.run()
    pi_res["Iteration to converge"].append(pi.iter)
    pi_res["Time to converge"].append(pi.time)
    pi_res["Max V"].append(pi.run_stats[-1]["Max V"])
    pi_res["Mean V"].append(pi.run_stats[-1]["Mean V"])
    V_all[("pi", s)] = pi.V
    policy_all[("pi", s)] = pi.policy
vi_df = da.DataFrame(vi_res, index=num_states)
pi_df = da.DataFrame(pi_res, index=num_states)
vi_df.to_csv("forest_val_nS.csv")
pi_df.to_csv("forest_pol_nS.csv")