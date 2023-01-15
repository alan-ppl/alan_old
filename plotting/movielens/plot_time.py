import numpy as np
import matplotlib.pyplot as plt
import json
from tueplots import axes, bundles

Ks = ['1','3','10','30']
Ns = ['30','200']
Ms = ['10','50','100']
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
            with open('results/movielens_results_local_IW_N{0}_M{1}.json'.format(N,M)) as f:
                results_local_IW = json.load(f)

            with open('results/movielens_results_alan_N{0}_M{1}.json'.format(N,M)) as f:
                results = json.load(f)

            # with open('results/movielens_tmc_results_alan_N{0}_M{1}.json'.format(N,M)) as f:
            #     results_tmc = json.load(f)

            with open('results/movielens_global_K_results_alan_N{0}_M{1}.json'.format(N,M)) as f:
                results_global_K = json.load(f)


            elbos_tpp = [results[N][M][k]['avg_time']/50000 for k in Ks]

            elbos_IW = [results_local_IW[N][M][k]['avg_time']/50000 for k in Ks]

            elbos_global_K = [results_global_K[N][M][k]['avg_time']/50000 for k in Ks]

            # elbos_tmc = [results_tmc[N][M][k]['avg_time']/50000 for k in Ks]


            ax[i,j].errorbar(Ks,elbos_IW, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='LIW')
            ax[i,j].errorbar(Ks,elbos_tpp, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Massively Parallel')
            ax[i,j].errorbar(Ks,elbos_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='Global K')
            # ax[i,j].errorbar(Ks,elbos_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='orange', label='TMC')
            #
            # ax.set_ylabel('Final Lower Bound')
            # ax.set_xlabel('K')
            # ax[i,j].label_outer()
            count =+ 1
    # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
    ax[0,0].set_title('Groups = 10')
    ax[0,1].set_title('Groups = 50')
    ax[0,2].set_title('Groups = 100')

    ax[0,0].set_ylabel('Obs per group = 30 \n Average time per iteration')
    ax[1,0].set_ylabel('Obs per group = 200 \n Average time per iteration')

    ax[1,0].sharex(ax[0,0])
    ax[1,0].set_xlabel('K')
    ax[1,1].sharex(ax[0,0])
    ax[1,1].set_xlabel('K')
    ax[1,2].sharex(ax[0,0])
    ax[1,2].set_xlabel('K')
    # fig.tight_layout()
    plt.legend()
    plt.savefig('charts/chart_time_movielens.png'.format(N, M))
    plt.savefig('charts/chart_time_movielens.pdf'.format(N, M))
