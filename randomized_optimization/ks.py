from pathlib import Path
_ROOT = Path(__file__).resolve().parent
_RESULTS = _ROOT / 'results'
_ASSETS = _ROOT / 'assets'
_DATA = _ROOT / 'data'
import timeit
import sys
sys.path.insert(0, str(_ROOT))
from _mlrose import require_mlrose
mlrose = require_mlrose()
from mlrose_hiive.generators import KnapsackGenerator
import matplotlib.pyplot as pt
import numpy as my
import random
random.seed(7)

"""
Knapsack: Given a set of items with a weight and value, determine the number of each item to include in a collection 
so that the total weight is less than or equal to a given limit and the total value is as large as possible
"""

iter_rng = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
location = str(_RESULTS / 'knapsack')
itm_typs = 35
mx_itms = 20

print('%s Item Type, %s Item Knapsack Value Optimization' %(itm_typs, mx_itms))
prb = KnapsackGenerator.generate(seed=7, number_of_items_types=itm_typs,
									max_item_count=mx_itms,max_weight_per_item=5,
									max_value_per_item=5,max_weight_pct=0.6)

def rndm_hc():
	rhc = mlrose.runners.RHCRunner(problem=prb,
						experiment_name='Knapsack RHC Optimal Params',
						output_directory=location,
						seed=7,
						iteration_list=2 ** my.arange(1,12),
						max_attempts=1000,
						restart_list=[0, 5, 25, 75])
	rhc_run_stts, rhc_run_crv = rhc.run()
	idl_rs = rhc_run_stts[['current_restart']].iloc[rhc_run_stts[['Fitness']].idxmax()]
	return rhc_run_stts, rhc_run_crv, idl_rs

rhc_df_run_stats, rhc_df_run_curves, ideal_rs = rndm_hc()
ideal_rs = int(ideal_rs['current_restart'])
print('RHC Ideal RS: ', ideal_rs)
rhc_bst_stt, rhc_bst_ftns, rhc_conv_tm = [], [], []
for k in iter_rng:
	strt = timeit.default_timer()
	bst_stt, bst_ftns, curve = mlrose.random_hill_climb(problem=prb, max_iters=k,
														max_attempts=1000, restarts=ideal_rs,
														curve=True)
	fin = timeit.default_timer()
	conv_tm = fin - strt
	rhc_bst_stt.append(bst_stt)
	rhc_bst_ftns.append(bst_ftns)
	rhc_conv_tm.append(conv_tm)

print('Fitness at the best state found via Random Hill Climbing: ', max(rhc_bst_ftns))

def gen_alg():
	ga = mlrose.runners.GARunner(problem=prb,
					  experiment_name='Knapsack GA Optimal Params',
					  output_directory=location,
					  seed=7,
					  iteration_list=2 ** my.arange(1,12),
					  max_attempts=1000,
					  population_sizes=[100, 300, 500],
					  mutation_rates=[0.2, 0.4, 0.6, 0.8, 1])
	ga_run_stts, ga_run_crv = ga.run()
	idl_pop_sz = ga_run_stts[['Population Size']].iloc[ga_run_stts[['Fitness']].idxmax()]
	idl_mutation_rate = ga_run_stts[['Mutation Rate']].iloc[ga_run_stts[['Fitness']].idxmax()]
	return ga_run_stts, ga_run_crv, idl_pop_sz, idl_mutation_rate

ga_df_run_stats, ga_df_run_curves, ideal_pop_size, ideal_mutation_rate = gen_alg()
ideal_pop_size = int(ideal_pop_size['Population Size'])
print('GA Ideal Pop: ', ideal_pop_size)
ideal_mutation_rate = float(ideal_mutation_rate['Mutation Rate'])
print('GA Ideal Mutation Rate: ', ideal_mutation_rate)
ga_bst_stt, ga_bst_ftns, ga_conv_tm = [], [], []
for k in iter_rng:
	strt = timeit.default_timer()
	bst_stt, bst_ftns, _ = mlrose.genetic_alg(problem=prb, curve=True,
												  mutation_prob = ideal_mutation_rate,
					      max_attempts = 1000, max_iters = k, pop_size=ideal_pop_size)
	fin = timeit.default_timer()
	conv_tm = fin - strt
	ga_bst_stt.append(bst_stt)
	ga_bst_ftns.append(bst_ftns)
	ga_conv_tm.append(conv_tm)

print('Fitness at the best route found via Genetic Algorithm: ', max(ga_bst_ftns))

def sim_ann():
	sa = mlrose.runners.SARunner(problem=prb,
				  experiment_name='Knapsack SA Optimal Params',
				  output_directory=location,
				  seed=7,
				  iteration_list=2 ** my.arange(12),
				  max_attempts=500,
				  temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
	sa_run_stts, sa_run_crv = sa.run()
	idl_temp = sa_run_stts[['Temperature']].iloc[sa_run_stts[['Fitness']].idxmax()]
	return sa_run_stts, sa_run_crv, idl_temp

sa_df_run_stats, sa_df_run_curves, ideal_initial_temp = sim_ann()
for i in ideal_initial_temp.Temperature:
    ideal_initial_temp = str(i)
    break
ideal_initial_temp = int(ideal_initial_temp)
print('SA Ideal Temp: ', ideal_initial_temp)
sa_bst_stt, sa_bst_ftns, sa_conv_tm = [], [], []
for k in iter_rng:
	strt = timeit.default_timer()
	bst_stt, bst_ftns, _ = mlrose.simulated_annealing(problem=prb, max_attempts = 500,
									max_iters = k, curve=True,
									schedule=mlrose.GeomDecay(init_temp=ideal_initial_temp))
	fin = timeit.default_timer()
	conv_tm = fin - strt
	sa_bst_stt.append(bst_stt)
	sa_bst_ftns.append(bst_ftns)
	sa_conv_tm.append(conv_tm)

print('Fitness at the best state found via Simulated Annealing: ', max(sa_bst_ftns))

def mimic():
	mmc = mlrose.runners.MIMICRunner(problem=prb,
				  experiment_name='Knapsack MIMIC Optimal Params',
				  output_directory=location,
				  seed=7,
				  iteration_list=2 ** my.arange(13),
				  max_attempts=1000, population_sizes=[100, 300, 500],
				  keep_percent_list=[0.25, 0.5], use_fast_mimic=True)
	mmc_run_stts, mmc_run_crv = mmc.run()
	idl_kp_prcnt = mmc_run_stts[['Keep Percent']].iloc[mmc_run_stts[['Fitness']].idxmax()]
	idl_pop_sz = mmc_run_stts[['Population Size']].iloc[mmc_run_stts[['Fitness']].idxmax()]
	return mmc_run_stts, mmc_run_crv, idl_kp_prcnt, idl_pop_sz

mmc_df_run_stats, mmc_df_run_curves, ideal_keep_prcnt, ideal_pop_size_mimic = mimic()
ideal_keep_prcnt = float(ideal_keep_prcnt['Keep Percent'])
print('M Ideal KP: ', ideal_keep_prcnt)
ideal_pop_size_mimic = int(ideal_pop_size_mimic['Population Size'])
print('M Ideal Pop Size: ', ideal_pop_size_mimic)
mimic_bst_stt, mimic_bst_ftns, mimic_conv_tm = [], [], []
for k in iter_rng:
	strt = timeit.default_timer()
	bst_stt, bst_ftns, _ = mlrose.mimic(problem=prb, keep_pct=ideal_keep_prcnt,
											max_attempts=1000, max_iters=k,
											pop_size=ideal_pop_size_mimic, curve=True)
	fin = timeit.default_timer()
	conv_tm = fin - strt
	mimic_bst_stt.append(bst_stt)
	mimic_bst_ftns.append(bst_ftns)
	mimic_conv_tm.append(conv_tm)

print('Fitness at the best state found via MIMIC: ', max(mimic_bst_ftns))

# Comparison Plots
fig, (ax1, ax2) = pt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(f'{mx_itms} Item Knapsack Random Search Optimizers Comparison: Fitness & Convergence Time')

ax1.set(xlabel="Iterations", ylabel="Fitness")
ax1.grid()
ax1.plot(iter_rng, rhc_bst_ftns, '-', color="g", label='Random Hill Climbing')
ax1.plot(iter_rng, sa_bst_ftns, '-', color="r", label='Simulated Annealing')
ax1.plot(iter_rng, ga_bst_ftns, '-', color="k", label='Genetic Algorithm')
ax1.plot(iter_rng, mimic_bst_ftns, '-', color="b", label='MIMIC')
ax1.legend(loc="best")

ax2.set(xlabel="Iterations [#]", ylabel="Convergence Time [s]")
ax2.grid()
ax2.plot(iter_rng, rhc_conv_tm, '-', color="g", label='Random Hill Climbing')
ax2.plot(iter_rng, sa_conv_tm, '-', color="r", label='Simulated Annealing')
ax2.plot(iter_rng, ga_conv_tm, '-', color="k", label='Genetic Algorithm')
ax2.plot(iter_rng, mimic_conv_tm, '-', color="b", label='MIMIC')
ax2.legend(loc="best")

pt.savefig('ks.png')
