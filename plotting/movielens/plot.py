import numpy as np
import matplotlib.pyplot as plt
import json
from tueplots import axes, bundles

Ks = ['1','3','10','30']
Ns = ['5','10']
Ms = ['10','50','300']
# with open('results.json') as f:
#     results = json.load(f)
#
# with open('results_local_IW.json') as f:
#     results_local_IW = json.load(f)

plt.rcParams.update({"figure.dpi": 300})
with plt.rc_context(bundles.icml2022()):

    count = 0
    fig, ax = plt.subplots(2,3, figsize=(5.5, 3.5))
    for i in range(len(Ns)):
        for j in range(len(Ms)):
            N = Ns[i]
            M = Ms[j]
            with open('results/movielens/elbo_N{0}_M{1}.json'.format(N,M)) as f:
                results = json.load(f)

            with open('results/movielens/elbo_LIW_N{0}_M{1}.json'.format(N,M)) as f:
                results_local_IW = json.load(f)

            with open('results/movielens/elbo_global_N{0}_M{1}.json'.format(N,M)) as f:
                results_global_K = json.load(f)

            # with open('results/results_tmc_lr_N{0}_M{1}.json'.format(N,M)) as f:
            #     results_tmc = json.load(f)


            elbos_tpp = [results[N][M][k]['final_obj'] for k in Ks]
            stds_tpp = [results[N][M][k]['final_obj_std']/np.sqrt(5) for k in Ks]

            elbos_IW = [results_local_IW[N][M][k]['final_obj'] for k in Ks]
            stds_IW = [results_local_IW[N][M][k]['final_obj_std']/np.sqrt(5) for k in Ks]

            elbos_global_K = [results_global_K[N][M][k]['final_obj'] for k in Ks]
            stds_global_K = [results_global_K[N][M][k]['final_obj_std']/np.sqrt(5) for k in Ks]

            # elbos_tmc = [results_tmc[N][M][k]['lower_bound'] for k in Ks]
            # stds_tmc = [results_tmc[N][M][k]['std']/np.sqrt(5) for k in Ks]


            ax[i,j].errorbar(Ks,elbos_IW, yerr=stds_IW, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='LIW')
            ax[i,j].errorbar(Ks,elbos_tpp, yerr=stds_tpp, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='TPP')
            ax[i,j].errorbar(Ks,elbos_global_K, yerr=stds_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='Global K')
            # ax[i,j].errorbar(Ks,elbos_tmc, yerr=stds_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='orange', label='TMC')
            #
            # ax.set_ylabel('Final Lower Bound')
            # ax.set_xlabel('K')
            # ax[i,j].label_outer()
            count =+ 1
    # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
    ax[0,0].set_title('Number of users = 10')
    ax[0,1].set_title('Number of users = 50')
    ax[0,2].set_title('Number of users = 300')

    ax[0,0].set_ylabel('Films per user = 5 \n Evidence Lower Bound')
    ax[1,0].set_ylabel('Films per user = 10 \n Evidence Lower Bound')

    ax[1,0].sharex(ax[0,0])
    ax[1,0].set_xlabel('K')
    ax[1,1].sharex(ax[0,0])
    ax[1,1].set_xlabel('K')
    ax[1,2].sharex(ax[0,0])
    ax[1,2].set_xlabel('K')
    # fig.tight_layout()
    plt.legend()
    plt.savefig('charts/chart_movielens.png'.format(N, M))
    plt.savefig('charts/chart_movielens.pdf'.format(N, M))
