import numpy as np
import matplotlib.pyplot as plt
import json
from tueplots import axes, bundles
from tueplots import cycler
from tueplots.constants import markers
from tueplots.constants.color import palettes

Ks_global = ['3','10','30','100','300', '1000', '3000', "10000", "30000"]
Ks_tmc = ['10']#,'10','30']
# Ns = ['5','10']
# Ms = ['50','150','300']
Ns = ['5']
Ms = ['300']
lrs = ['0.15']# '0.01', '0.0001', '1e-05', '1e-06']
# with open('results.json') as f:
#     results = json.load(f)
#
# with open('results_local_IW.json') as f:
#     results_local_IW = json.load(f)


plt.rcParams.update({"figure.dpi": 300})
plt.rcParams.update(cycler.cycler(color=palettes.muted))
with plt.rc_context(bundles.icml2022()):

    count = 0
    fig, ax = plt.subplots(1,1, figsize=(5.5/2, 2.0))
    for i in range(len(Ns)):
        for j in range(len(Ms)):
            for lr_ml in range(len(lrs)):
                N = Ns[i]
                M = Ms[j]
    #            colour =
                lr_ml = lrs[lr_ml]
                for K in ['30', '100']:
                    with open('results/movielens/ML_1_{}__N{}_M{}_K{}.json'.format(lr_ml,N,M,K)) as f:
                        results_ml_tmc_new = json.load(f)




                    elbos_ml_tmc_new = results_ml_tmc_new[N][M][K]['pred_likelihood']
                    stds_ml_tmc_new = results_ml_tmc_new[N][M][K]['pred_likelihood_std']
                    time_ml_tmc_new = results_ml_tmc_new[N][M][K]['time']
                    ax.errorbar(time_ml_tmc_new,elbos_ml_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', label='ML lr: {} K:{}'.format(lr_ml, K))

                for lr in ['0.1']:
                    with open('results/movielens/VI_1_{}__N{}_M{}_K30.json'.format(lr,N,M)) as f:
                        results_adam_tmc_new = json.load(f)

                    elbos_adam_tmc_new = results_adam_tmc_new[N][M]['30']['pred_likelihood']
                    stds_adam_tmc_new = results_adam_tmc_new[N][M]['30']['pred_likelihood_std']
                    time_adam_tmc_new = results_adam_tmc_new[N][M]['30']['time']

                    ax.errorbar(time_adam_tmc_new,elbos_adam_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', label='MP VI lr: {} K:30'.format(lr))

                count =+ 1
    ax.set_title('ML vs MP VI')
    ax.set_ylabel('Predictive Log Likelihood')

    ax.set_xlabel('Time (Seconds)')

    plt.legend()
    plt.savefig('charts/chart_movielens_ml1_predll_N{}_M{}.png'.format(N, M))
    plt.savefig('charts/chart_movielens_ml1_predll_N{}_M{}.pdf'.format(N, M))


with plt.rc_context(bundles.icml2022()):

    count = 0
    fig, ax = plt.subplots(1,1, figsize=(5.5/2, 2.0))
    for i in range(len(Ns)):
        for j in range(len(Ms)):
            for lr_ml in range(len(lrs)):
                N = Ns[i]
                M = Ms[j]
    #            colour =
                lr_ml = lrs[lr_ml]

                for K in ['30', '100']:
                    with open('results/movielens/ML_1_{}__N{}_M{}_K{}.json'.format(lr_ml,N,M,K)) as f:
                        results_ml_tmc_new = json.load(f)


                    elbos_ml_tmc_new = results_ml_tmc_new[N][M][K]['objs']
                    time_ml_tmc_new = results_ml_tmc_new[N][M][K]['time']
                    ax.errorbar(np.arange(200),elbos_ml_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', label='ML lr: {} K:{}'.format(lr_ml, K))

                for lr in ['0.1']:
                    with open('results/movielens/VI_1_{}__N{}_M{}_K30.json'.format(lr,N,M)) as f:
                        results_adam_tmc_new = json.load(f)
                    elbos_adam_tmc_new = results_adam_tmc_new[N][M]['30']['objs']
                    time_adam_tmc_new = results_adam_tmc_new[N][M]['30']['time']

                    ax.errorbar(np.arange(200),elbos_adam_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', label='MP VI lr: {} K:30'.format(lr))

                count =+ 1

    ax.set_title('ML vs MP VI')
    ax.set_ylabel('Elbo')

    ax.set_xlabel('Iterations')

    plt.legend()
    plt.savefig('charts/chart_movielens_ml1_elbo_N{}_M{}.png'.format(N, M))
    plt.savefig('charts/chart_movielens_ml1_elbo_N{}_M{}.pdf'.format(N, M))

with plt.rc_context(bundles.icml2022()):

    count = 0
    fig, ax = plt.subplots(1,1, figsize=(5.5/2, 2.0))
    for i in range(len(Ns)):
        for j in range(len(Ms)):
            for lr_ml in range(len(lrs)):
                N = Ns[i]
                M = Ms[j]
    #            colour =
                lr_ml = lrs[lr_ml]

                for K in ['30', '100']:
                    with open('results/movielens/ML_1_{}__N{}_M{}_K{}.json'.format(lr_ml,N,M,K)) as f:
                        results_ml_tmc_new = json.load(f)


                    elbos_ml_tmc_new = results_ml_tmc_new[N][M][K]['objs']
                    time_ml_tmc_new = results_ml_tmc_new[N][M][K]['time']

                    ax.errorbar(time_ml_tmc_new,elbos_ml_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', label='ML lr: {} K:{}'.format(lr_ml, K))

                for lr in ['0.1']:
                    with open('results/movielens/VI_1_{}__N{}_M{}_K30.json'.format(lr,N,M)) as f:
                        results_adam_tmc_new = json.load(f)
                    elbos_adam_tmc_new = results_adam_tmc_new[N][M]['30']['objs']
                    time_adam_tmc_new = results_adam_tmc_new[N][M]['30']['time']

                    ax.errorbar(time_adam_tmc_new,elbos_adam_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', label='MP VI lr: {} K:30'.format(lr))

                count =+ 1

    ax.set_title('ML vs MP VI')
    ax.set_ylabel('Elbo')

    ax.set_xlabel('Time (Seconds)')

    plt.legend()
    plt.savefig('charts/chart_time_movielens_ml1_elbo_N{}_M{}.png'.format(N, M))
    plt.savefig('charts/chart_time_movielens_ml1_elbo_N{}_M{}.pdf'.format(N, M))
