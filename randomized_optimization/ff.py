from pathlib import Path
_ROOT = Path(__file__).resolve().parent
_RESULTS = _ROOT / 'results'
_ASSETS = _ROOT / 'assets'
_DATA = _ROOT / 'data'
import timeit
try:
    import mlrose_hiive as mlrose
except ImportError:  # pragma: no cover
    import mlrose
import matplotlib.pyplot as pt
import numpy as my
try:
    from mlrose_hiive.generators import FlipFlopGenerator
except ImportError:  # pragma: no cover
    from mlrose.generators import FlipFlopGenerator
import random
random.seed(7)

"""
Flipflop: counts the number of bit flips in an array of 0s and 1s with the goal (max) is n-1 flips.
"""

location = str(_RESULTS / 'flipflop')
iter_rng = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
bts = 50

print(f'{bts} Bit FlipFlop Optimization')
prb = FlipFlopGenerator.generate(seed=7, size=bts)

def rndm_hc():
	rhc = mlrose.runners.RHCRunner(problem=prb,
						experiment_name='Flipflop RHC Optimal Params',
						output_directory=location,
						seed=7,
						iteration_list=2 ** my.arange(1,12),
						max_attempts=5000,
						restart_list=[0, 5, 25, 75])
	rhc_run_stts, rhc_run_crvs = rhc.run()
	idl_rs = rhc_run_stts[['current_restart']].iloc[rhc_run_stts[['Fitness']].idxmax()]
	return rhc_run_stts, rhc_run_crvs, idl_rs

rhc_df_run_stats, rhc_df_run_curves, ideal_rs = rndm_hc()
ideal_rs = int(ideal_rs['current_restart'])
print('RHC Ideal RS: ', ideal_rs)
rhc_bst_stt, rhc_bst_ftns, rhc_conv_tm = [], [], []

for k in iter_rng:
	strt = timeit.default_timer()
	bst_stt, bst_ftns, curve = mlrose.random_hill_climb(problem=prb, max_iters=k, max_attempts=5000, restarts=ideal_rs, curve=True, random_state=7)
	fin = timeit.default_timer()
	conv_tm = fin - strt
	rhc_bst_stt.append(bst_stt)
	rhc_bst_ftns.append(bst_ftns)
	rhc_conv_tm.append(conv_tm)

print('Fitness at the best state via Random Hill Climbing: ', max(rhc_bst_ftns))

def gen_alg():
	ga = mlrose.runners.GARunner(problem=prb,
					  experiment_name='Flipflop GA Optimal Params',
					  output_directory=location,
					  seed=7,
					  iteration_list=2 ** my.arange(1,12),
					  max_attempts=5000,
					  population_sizes=[10, 50, 100],
					  mutation_rates=[0.2, 0.4, 0.6, 0.8, 1])
	ga_run_stts, ga_run_crv = ga.run()
	idl_pop_sz = ga_run_stts[['Population Size']].iloc[ga_run_stts[['Fitness']].idxmax()]
	idl_mut_rt = ga_run_stts[['Mutation Rate']].iloc[ga_run_stts[['Fitness']].idxmax()]
	return ga_run_stts, ga_run_crv, idl_pop_sz, idl_mut_rt

ga_df_run_stats, ga_df_run_curves, ideal_pop_size, ideal_mutation_rate = gen_alg()
ideal_pop_size = int(ideal_pop_size['Population Size'])
print('GA Ideal Pop: ', ideal_pop_size)
ideal_mutation_rate = float(ideal_mutation_rate['Mutation Rate'])
print('GA Ideal Mutation Rate: ', ideal_mutation_rate)
ga_bst_stt, ga_bst_ftns, ga_conv_tm = [], [], []
for k in iter_rng:
	strt = timeit.default_timer()
	bst_stt, bst_ftns, _ = mlrose.genetic_alg(problem=prb,
							mutation_prob = ideal_mutation_rate, curve=True,
					      	max_attempts = 5000, random_state=7, max_iters = k, pop_size=ideal_pop_size)
	fin = timeit.default_timer()
	conv_tm = fin - strt
	ga_bst_stt.append(bst_stt)
	ga_bst_ftns.append(bst_ftns)
	ga_conv_tm.append(conv_tm)

print('Fitness at the best route via Genetic Algorithms: ', max(ga_bst_ftns))

def sim_ann():
	sa = mlrose.runners.SARunner(problem=prb,
				  experiment_name='Flipflop SA Optimal Params',
				  output_directory=location,
				  seed=7,
				  iteration_list=2 ** my.arange(12),
				  max_attempts=5000,
				  temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
	sa_run_stts, sa_run_crv = sa.run()
	idl_tmp = sa_run_stts[['Temperature']].iloc[sa_run_stts[['Fitness']].idxmax()]
	return sa_run_stts, sa_run_crv, idl_tmp

sa_df_run_stats, sa_df_run_curves, ideal_initial_temp = sim_ann()
for i in ideal_initial_temp.Temperature:
    ideal_initial_temp = str(i)
    break
ideal_initial_temp = int(ideal_initial_temp)
print('SA Ideal Temp: ', ideal_initial_temp)
sa_bst_stt, sa_bst_ftns, sa_conv_tm = [], [], []

for k in iter_rng:
	strt = timeit.default_timer()
	bst_stt, bst_ftns, _ = mlrose.simulated_annealing(problem=prb, max_attempts = 5000,
									max_iters = k, random_state=7, curve=True,
									schedule=mlrose.GeomDecay(init_temp=ideal_initial_temp))
	fin = timeit.default_timer()
	conv_tm = fin - strt
	sa_bst_stt.append(bst_stt)
	sa_bst_ftns.append(bst_ftns)
	sa_conv_tm.append(conv_tm)

print('Fitness at the best state via Simulated Annealing: ', max(sa_bst_ftns))

def mimic():
	mmc = mlrose.runners.MIMICRunner(problem=prb,
				  experiment_name='Flipflop MIMIC Optimal Params',
				  output_directory=location,
				  seed=7,
				  iteration_list=2 ** my.arange(13),
				  max_attempts=5000, population_sizes=[100, 300, 500],
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
	bst_stt, bst_ftns, _ = mlrose.mimic(problem=prb, keep_pct=ideal_keep_prcnt, random_state=7,
											max_attempts=5000, max_iters=k,
											pop_size=ideal_pop_size_mimic, curve=True)
	fin = timeit.default_timer()
	conv_tm = fin - strt
	mimic_bst_stt.append(bst_stt)
	mimic_bst_ftns.append(bst_ftns)
	mimic_conv_tm.append(conv_tm)

print('Fitness at the best state via MIMIC: ', max(mimic_bst_ftns))

# Comparison Plots
fig, (ax1, ax2) = pt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(f'Flipflop Random Search Optimizers Comparison: Fitness & Convergence Time {bts}')

ax1.set(xlabel="Iterations [#]", ylabel="Fitness")
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

pt.savefig('ff.png')