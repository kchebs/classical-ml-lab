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
import matplotlib
from sklearn.model_selection import cross_validate, train_test_split
import numpy as my
import pandas as ds
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as pt
import random
import warnings
random.seed(7)
from sklearn import preprocessing

hid_nds = [5]  # Confirmed by exhaustive NN grid search in supervised_learning

bc_df = ds.read_csv(str(_DATA / "breastcancer.csv"))
print("Dataset size: ",len(bc_df),"instances and", len(bc_df.columns) - 1,"features")
co = list(bc_df)
co.insert(0, co.pop(co.index("diagnosis")))
bc_df["diagnosis"].replace("M",1,inplace=True)
bc_df = bc_df.loc[:, co]
bc_df["diagnosis"].replace("B",0,inplace=True)
bc_df["diagnosis"] = bc_df["diagnosis"].astype("category")

bcx = my.array(bc_df.values[:,1:],dtype="int64")
bcx = preprocessing.scale(bcx)
bcy = my.array(bc_df.values[:,0],dtype="int64")

bc_xtn, bc_xtt, bc_ytn, bc_ytt = train_test_split(my.array(bcx),my.array(bcy), test_size=0.20, random_state=7)
trgts = ['Benign', 'Malignant']
ftrs = bc_df.columns.values[1:]

def bld_mdl(algo, hddn_nds, actvtn, schdl=mlrose.GeomDecay(init_temp=5000), rstrts=75,
                pop=300, mut=0.2, mx_iter=5000, lr=0.001):
    return mlrose.neural.NeuralNetwork(hidden_nodes=hddn_nds, activation=actvtn,
                                        algorithm=algo, max_iters=mx_iter,
                                        bias=True, is_classifier=True, learning_rate=lr,
                                        early_stopping=False, restarts = rstrts, clip_max=1,
                                        schedule=schdl,
                                        max_attempts=1000, pop_size=pop, mutation_prob=mut,
                                        random_state=7, curve=True)

def lrn_crv(estimator, algo, dataset, xtrn, ytrn, cv=5):
    """
    Creates test and training learning curves

    Parameters
    ----------
    estimator : object type that enacts the predict and fit methods

    algo : string of search algorithm for chart title

    dataset : string of dataset for chart title

    xtrn : array-like training vector of shape (n_samples, n_features)

    ytrn : array-like target vector relative to xtrn

    cv : int, an optional iterable or cross-validation generator with 5 as the default
    """
    warnings.filterwarnings('ignore', 'F-score is ill-defined.*')

    trn_scr_avg = []; trn_scr_std = []
    tst_scr_avg = []; tst_scr_std = []
    trn_tm_avg = []; trn_tm_std = []
    tst_tm_avg =[]; tst_tm_std = []

    trn_sz = (my.linspace(.25, 1.0, cv) * len(ytrn)).astype('int')
    for k in trn_sz:
        index = my.random.randint(xtrn.shape[0], size = k)
        cvr = cross_validate(estimator, xtrn[index,:], ytrn[index], cv=cv, scoring='f1_weighted', n_jobs=-1, return_train_score=True)

        trn_scr_avg.append(my.mean(cvr['train_score'])); trn_scr_std.append(my.std(cvr['train_score']))
        tst_scr_avg.append(my.mean(cvr['test_score'])); tst_scr_std.append(my.std(cvr['test_score']))
        trn_tm_avg.append(my.mean(cvr['fit_time'])); trn_tm_std.append(my.std(cvr['fit_time']))
        tst_tm_avg.append(my.mean(cvr['score_time'])); tst_tm_std.append(my.std(cvr['score_time']))

    # Convert arrays to numpy arrays for later math operations
    trn_scr_avg = my.array(trn_scr_avg); trn_scr_std = my.array(trn_scr_std)
    tst_scr_avg = my.array(tst_scr_avg); tst_scr_std = my.array(tst_scr_std)
    trn_tm_avg = my.array(trn_tm_avg); trn_tm_std = my.array(trn_tm_std)
    tst_tm_avg = my.array(tst_tm_avg); tst_tm_std = my.array(tst_tm_std)

    fig, (ax1, ax2) = pt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(algo + " Learning Curves for: " + dataset)
    ax1.set(xlabel="Training Ex [#]", ylabel="Model F1 Score")

    # F1 score (y) vs training size (X) Learning Curve
    ax1.grid()
    ax1.fill_between(trn_sz, trn_scr_avg - trn_scr_std,
                     trn_scr_avg + trn_scr_std, alpha=0.1,
                     color="r")
    ax1.fill_between(trn_sz, tst_scr_avg - tst_scr_std,
                     tst_scr_avg + tst_scr_std, alpha=0.1, color="g")
    ax1.plot(trn_sz, trn_scr_avg, '-', color="r",
             label="Training")
    ax1.plot(trn_sz, tst_scr_avg, '-', color="g",
             label="Testing")
    ax1.legend(loc="best")

    # Time (y) vs training size (X) Learning Curve
    ax2.set(xlabel="Training Ex [#]", ylabel="Model Time [s]")
    ax2.grid()
    ax2.fill_between(trn_sz, trn_tm_avg - trn_tm_std,
                     trn_tm_avg + trn_tm_std, alpha=0.1,
                     color="r")
    ax2.fill_between(trn_sz, tst_tm_avg - tst_tm_std,
                     tst_tm_avg + tst_tm_std, alpha=0.1, color="g")
    ax2.plot(trn_sz, trn_tm_avg, '-', color="r",
             label="Training [s]")
    ax2.plot(trn_sz, tst_tm_avg, '-', color="g",
             label="Prediction [s]")
    ax2.legend(loc="best")

    pt.savefig(f'{algo}_lc.png')

    return trn_sz, trn_scr_avg, trn_tm_avg, tst_tm_avg


def eval_mdl(nn_model, algo, xtrn, xtst, ytrn, ytst, class_names):
    # Training
    strt = timeit.default_timer()
    nn_model.fit(X=xtrn, y=ytrn)
    fin = timeit.default_timer()
    trn_tm = fin - strt
    nn_ft_crv = nn_model.fitness_curve

    # Prediction
    strt_p = timeit.default_timer()
    yprd = nn_model.predict(xtst)
    fin_p = timeit.default_timer()
    prd_tm = fin_p - strt_p

    # Prediction Distribution
    decdd = []
    for k in range(0, len(yprd)):
        if yprd[k] == [0]:
            decdd.insert(k, class_names[0])
        elif yprd[k] == [1]:
            decdd.insert(k, class_names[1])

    unq, cnts = my.unique(decdd, return_counts=True)
    res_classes = dict(zip(unq, cnts))
    fig, ax = pt.subplots()
    pt.bar(*zip(*res_classes.items()))
    pt.xticks((0, 1), labels=class_names)
    ax.title.set_text("Prediction Distribution over all Classes")
    pt.savefig(f'{algo}_eval_mdl.png')

    f1 = f1_score(ytst, yprd, average='binary', labels=my.unique(ytst))
    acc = accuracy_score(ytst, yprd)
    prc = precision_score(ytst, yprd, average='binary', labels=my.unique(ytst))
    rcll = recall_score(ytst, yprd, average='binary', labels=my.unique(ytst))

    print(f"{algo} Search Algorithm Neural Network Metrics")
    print("Training Time (ms):   " + "{:.4f}".format(trn_tm))
    print("Prediction Time (ms): " + "{:.4f}\n".format(prd_tm))
    print("F1 Score:  " + "{:.3f}".format(f1))
    print("Accuracy:  " + "{:.3f}".format(acc))
    print("Recall:    " + "{:.3f}".format(rcll))
    print("Precision: " + "{:.3f}".format(prc))
    return f1, acc, prc, rcll, trn_tm, prd_tm, nn_ft_crv

# Gradient Descent
gd = bld_mdl(algo='gradient_descent', hddn_nds=hid_nds, actvtn='relu',
                          mx_iter=5000, lr=0.001)

gd_contra_trn_sz, gd_contra_trn_scr_avg, gd_contra_trn_tm_avg, gd_contra_tst_tm_avg = \
     lrn_crv(estimator=gd, algo='Gradient Descent', dataset="Breast Cancer", xtrn=bc_xtn, ytrn=bc_ytn, cv=3)

gd_nn_test_f1, gd_nn_test_acc, gd_nn_test_precision, gd_nn_test_recall, gd_nn_trn_tm, gd_nn_tst_tm, nngd_ft = \
    eval_mdl(nn_model=gd, algo='Gradient Descent', xtrn=bc_xtn, xtst=bc_xtt, ytrn=bc_ytn,
             ytst=bc_ytt, class_names=trgts)

# Random Hill Climbing
rhc = bld_mdl(algo='random_hill_climb', hddn_nds=hid_nds, actvtn='relu',
                           rstrts=75, mx_iter=5000, lr=0.001)

rhc_contra_trn_sz, rhc_contra_trn_scr_avg, rhc_contra_trn_tm_avg, rhc_contra_tst_tm_avg = \
    lrn_crv(estimator=rhc, algo='RHC', dataset="Breast Cancer", xtrn=bc_xtn, ytrn=bc_ytn, cv=3)

rhc_nn_test_f1, rhc_nn_test_acc, rhc_nn_test_precision, rhc_nn_test_recall, rhc_nn_trn_tm, rhc_nn_tst_tm, nnrhc_ft = \
    eval_mdl(nn_model=rhc, algo='RHC', xtrn=bc_xtn, xtst=bc_xtt, ytrn=bc_ytn,
             ytst=bc_ytt, class_names=trgts)

# Genetic Algorithms
ga = bld_mdl(algo='genetic_alg', hddn_nds=hid_nds, actvtn='relu',
                          pop=300, mut=0.2, mx_iter=5000, lr=0.001)

ga_contra_trn_sz, ga_contra_trn_scr_avg, ga_contra_trn_tm_avg, ga_contra_tst_tm_avg = \
    lrn_crv(estimator=ga, algo='GA', dataset="Breast Cancer", xtrn=bc_xtn, ytrn=bc_ytn, cv=3)

ga_nn_test_f1, ga_nn_test_acc, ga_nn_test_precision, ga_nn_test_recall, ga_nn_trn_tm, ga_nn_tst_tm, nnga_ft = \
    eval_mdl(nn_model=ga, algo='GA', xtrn=bc_xtn, xtst=bc_xtt, ytrn=bc_ytn,
             ytst=bc_ytt, class_names=trgts)

# Simulated Annealing
sa = bld_mdl(algo='simulated_annealing', hddn_nds=hid_nds, actvtn='relu', schdl=mlrose.algorithms.decay.ArithDecay(init_temp=5000),
                          mx_iter=5000, lr=0.001)

sa_contra_trn_sz, sa_contra_trn_scr_avg, sa_contra_trn_tm_avg, sa_contra_tst_tm_avg = \
    lrn_crv(estimator=sa, algo='SA', dataset="Breast Cancer", xtrn=bc_xtn, ytrn=bc_ytn, cv=3)
sa_nn_test_f1, sa_nn_test_acc, sa_nn_test_precision, sa_nn_test_recall, sa_nn_trn_tm, sa_nn_tst_tm, nnsa_ft = \
    eval_mdl(nn_model=sa, algo='SA', xtrn=bc_xtn, xtst=bc_xtt, ytrn=bc_ytn,
             ytst=bc_ytt, class_names=trgts)

nngd_ft *= -1


# Comparison Plots
fig1, (ax3, ax4) = pt.subplots(1, 2, figsize=(15, 5))
fig1.suptitle('Random Search Optimizers Comparison on Neural Network Weight Optimization: F1-Score and Convergence Time')

ax3.set(xlabel="Iterations [#]", ylabel="Loss")
ax3.grid()
ax3.plot(nnrhc_ft, '-', color="g",label="RHC Training Fitness")
ax3.plot(nnsa_ft, '-', color="r",label="SA Training Fitness")
ax3.plot(nnga_ft, '-', color="k",label="GA Training Fitness")
ax3.plot(nngd_ft, '-', color="b",label="Gradient Descent Training Fitness")
ax3.legend(loc="best")

ax4.set(xlabel="Iterations [#]", ylabel="Training Time [s]")
ax4.grid()
ax4.plot(rhc_nn_trn_tm, '-', color="g", label='RHC')
ax4.plot(sa_nn_trn_tm, '-', color="r", label='SA')
ax4.plot(ga_nn_trn_tm, '-', color="k", label='GA')
ax4.plot(gd_nn_trn_tm, '-', color="b", label='Gradient Descent')
ax4.legend(loc="best")

pt.savefig('nn.png')

