import numpy as np
import matplotlib.pyplot as plt
import json
from tueplots import axes, bundles

Ks = ['1','5','10','15']
Ns = ['200','30']
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

            with open('results/movielens_results_rws_N{0}_M{1}.json'.format(N,M)) as f:
                results_rws = json.load(f)

            with open('results/movielens_results_tpp_N{0}_M{1}.json'.format(N,M)) as f:
                results_tpp = json.load(f)



            elbos_rws = [results_rws[N][M][k]['lower_bound'] for k in Ks]
            stds_rws = [results_rws[N][M][k]['std']/np.sqrt(5) for k in Ks]

            elbos_rws[0] = np.nan
            stds_rws[0] = 0

            elbos_IW = [results_local_IW[N][M][k]['lower_bound'] for k in Ks]
            stds_IW = [results_local_IW[N][M][k]['std']/np.sqrt(5) for k in Ks]

            elbos_tpp = [results_tpp[N][M][k]['lower_bound'] for k in Ks]
            stds_tpp = [results_tpp[N][M][k]['std']/np.sqrt(5) for k in Ks]

            ax[i,j].errorbar(Ks,elbos_IW, yerr=stds_IW, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='LIW')
            # ax[i,j].errorbar(Ks,elbos_rws, yerr=stds_rws, linewidth=0.55, markersize = 0.75, fmt='-o', c='orange', label='RWS')
            ax[i,j].errorbar(Ks,elbos_tpp, yerr=stds_tpp, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='TPP')
            #
            # ax.set_ylabel('Final Lower Bound')
            # ax.set_xlabel('K')
            # ax[i,j].label_outer()
            count =+ 1
    # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
    ax[0,0].set_title('Groups = 10')
    ax[0,1].set_title('Groups = 50')
    ax[0,2].set_title('Groups = 100')

    ax[0,0].set_ylabel('Obs per group = 30 \n Final Lower Bound')
    ax[1,0].set_ylabel('Obs per group = 200 \n Final Lower Bound')

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
