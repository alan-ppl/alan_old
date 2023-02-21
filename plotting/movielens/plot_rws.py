import numpy as np
import matplotlib.pyplot as plt
import json
from tueplots import axes, bundles

Ks_global = ['3','10','30','100','300', '1000', '3000', "10000", "30000"]
Ks_tmc = ['3','10','30']
# Ns = ['5','10']
# Ms = ['50','150','300']
Ns = ['100']
Ms = ['10']
# with open('results.json') as f:
#     results = json.load(f)
#
# with open('results_local_IW.json') as f:
#     results_local_IW = json.load(f)

plt.rcParams.update({"figure.dpi": 300})
with plt.rc_context(bundles.icml2022()):

    count = 0
    fig, ax = plt.subplots(1,1, figsize=(5.5/2, 2.0))
    for i in range(len(Ns)):
        for j in range(len(Ms)):
            N = Ns[i]
            M = Ms[j]

            with open('results/movielens_discrete/rws_global_N{0}_M{1}.json'.format(N,M)) as f:
                results_rws_global_k = json.load(f)

            with open('results/movielens_discrete/rws_tmc_N{0}_M{1}.json'.format(N,M)) as f:
                results_rws_tmc = json.load(f)

            with open('results/movielens_discrete/rws_tmc_new_N{0}_M{1}.json'.format(N,M)) as f:
                results_rws_tmc_new = json.load(f)

            with open('results/movielens_discrete/rws_tmc_new_LIW_N{0}_M{1}.json'.format(N,M)) as f:
                results_IW = json.load(f)



            # elbos_rws_tmc = [results_rws_tmc[N][M][k]['pred_likelihood'] for k in Ks]
            # stds_rws_tmc = [results_rws_tmc[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]

            elbos_rws_tmc_new = [results_rws_tmc_new[N][M][k]['pred_likelihood'] for k in Ks_tmc] + [np.nan]*6
            stds_rws_tmc_new = [results_rws_tmc_new[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks_tmc] + [np.nan]*6

            elbos_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood'] for k in Ks_global]
            stds_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks_global]

            # elbos_rws_IW = [results_IW[N][M][k]['pred_likelihood'] for k in Ks]
            # stds_rws_IW = [results_IW[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]

            ax.errorbar(Ks_global,elbos_rws_global_k, yerr=stds_rws_global_k, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global RWS')
            # ax[i,j].errorbar(Ks,elbos_rws_tmc, yerr=stds_rws_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='orange', label='TMC RWS')
            ax.errorbar(Ks_global,elbos_rws_tmc_new, yerr=stds_rws_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP RWS')
            # ax[i,j].errorbar(Ks,elbos_rws_IW, yerr=stds_rws_IW, linewidth=0.55, markersize = 0.75, fmt='-o', c='purple', label='RWS LIW')

            count =+ 1
    # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
    # ax[0,0].set_title('Number of users = 50')
    # ax[0,1].set_title('Number of users = 150')
    ax.set_title('Number of users = 300')

    ax.set_ylabel('Films per user = 5 \n Predictive Log Likelihood')
    # ax[1,0].set_ylabel('Films per user = 10 \n Predictive Log Likelihood')

    # ax[1,0].sharex(ax[0,0])
    ax.set_xlabel('K')
    ax.annotate(r'\bf{a}', xy=(0, 1), xycoords='axes fraction', fontsize=10,
                xytext=(-40, 5), textcoords='offset points',
                ha='right', va='bottom')
    # ax[1,1].sharex(ax[0,0])
    # ax[1,1].set_xlabel('K')
    # ax[1,2].sharex(ax[0,0])
    # ax[1,2].set_xlabel('K')
    # fig.tight_layout()
    # plt.legend()

    plt.savefig('charts/chart_movielens_rwsN5M300.png'.format(N, M))
    plt.savefig('charts/chart_movielens_rwsN5M300.pdf'.format(N, M))


plt.rcParams.update({"figure.dpi": 300})
with plt.rc_context(bundles.icml2022()):

    count = 0
    fig, ax = plt.subplots(1,1, figsize=(5.5/2, 2.0))
    for i in range(len(Ns)):
        for j in range(len(Ms)):
            N = Ns[i]
            M = Ms[j]

            with open('results/movielens_discrete/rws_global_N{0}_M{1}.json'.format(N,M)) as f:
                results_rws_global_k = json.load(f)

            with open('results/movielens_discrete/rws_tmc_N{0}_M{1}.json'.format(N,M)) as f:
                results_rws_tmc = json.load(f)

            with open('results/movielens_discrete/rws_tmc_new_N{0}_M{1}.json'.format(N,M)) as f:
                results_rws_tmc_new = json.load(f)

            with open('results/movielens_discrete/rws_tmc_new_LIW_N{0}_M{1}.json'.format(N,M)) as f:
                results_IW = json.load(f)



            elbos_rws_tmc_new = [results_rws_tmc_new[N][M][k]['pred_likelihood'] for k in Ks_tmc] + [np.nan]*6
            stds_rws_tmc_new = [results_rws_tmc_new[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks_tmc] + [np.nan]*6
            time_rws_tmc_new = [results_rws_tmc_new[N][M][k]['avg_time'] for k in Ks_tmc] + [np.nan]*6
            stds_rws_tmc_new_time = [results_rws_tmc_new[N][M][k]['std_time']/np.sqrt(5) for k in Ks_tmc] + [np.nan]*6


            elbos_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood'] for k in Ks_global]
            stds_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks_global]
            time_rws_global_k = [results_rws_global_k[N][M][k]['avg_time'] for k in Ks_global]
            stds_rws_global_k_time = [results_rws_global_k[N][M][k]['std_time']/np.sqrt(5) for k in Ks_global]



            ax.errorbar(time_rws_global_k,elbos_rws_global_k, xerr=stds_rws_global_k_time, yerr=stds_rws_global_k, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global RWS')
            # ax[i,j].errorbar(Ks,elbos_rws_tmc, yerr=stds_rws_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='orange', label='TMC RWS')
            ax.errorbar(time_rws_tmc_new,elbos_rws_tmc_new, xerr=stds_rws_tmc_new_time, yerr=stds_rws_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP RWS')
            # ax[i,j].errorbar(Ks,elbos_rws_IW, yerr=stds_rws_IW, linewidth=0.55, markersize = 0.75, fmt='-o', c='purple', label='RWS LIW')

            count =+ 1
    # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
    # ax[0,0].set_title('Number of users = 50')
    # ax[0,1].set_title('Number of users = 150')
    ax.set_title('Number of users = 300')

    ax.set_ylabel('Films per user = 5 \n Predictive Log Likelihood')
    # ax[1,0].set_ylabel('Films per user = 10 \n Predictive Log Likelihood')

    # ax[1,0].sharex(ax[0,0])
    ax.set_xlabel('Time (Seconds)')
    ax.annotate(r'\bf{b}', xy=(0, 1), xycoords='axes fraction', fontsize=10,
                xytext=(-40, 5), textcoords='offset points',
                ha='right', va='bottom')
    # ax[1,1].sharex(ax[0,0])
    # ax[1,1].set_xlabel('K')
    # ax[1,2].sharex(ax[0,0])
    # ax[1,2].set_xlabel('K')
    # fig.tight_layout()
    plt.legend()
    plt.savefig('charts/chart_movielens_rws_timeN5M300.png'.format(N, M))
    plt.savefig('charts/chart_movielens_rws_timeN5M300.pdf'.format(N, M))
