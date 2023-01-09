import numpy as np
import matplotlib.pyplot as plt
import json
from tueplots import axes, bundles

Ks = ['3','10','30']
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

            with open('results/rws_global_N{0}_M{1}.json'.format(N,M)) as f:
                results_rws_global_k = json.load(f)

            with open('results/rws_N{0}_M{1}.json'.format(N,M)) as f:
                results_rws = json.load(f)

            # with open('results/movielens_results_tmc_rws_N{0}_M{1}.json'.format(N,M)) as f:
            #     results_rws_tmc = json.load(f)



            elbos_rws = [results_rws[N][M][k]['pred_likelihood'] for k in Ks]
            stds_rws = [results_rws[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]


            elbos_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood'] for k in Ks]
            stds_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]

            # elbos_rws_tmc = [results_rws_tmc[N][M][k]['pred_mean'] for k in Ks]
            # stds_rws_tmc = [results_rws_tmc[N][M][k]['pred_std']/np.sqrt(5) for k in Ks]
            #
            # print(elbos_rws)
            # print(elbos_rws_global_k)
            ax[i,j].errorbar(Ks,elbos_rws_global_k, yerr=stds_rws_global_k, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='Global K RWS')
            ax[i,j].errorbar(Ks,elbos_rws, yerr=stds_rws, linewidth=0.55, markersize = 0.75, fmt='-o', c='orange', label='TPP RWS')
            # ax[i,j].errorbar(Ks,elbos_rws_tmc, yerr=stds_rws_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='purple', label='TMC RWS')
            # ax.set_ylabel('Final Lower Bound')
            # ax.set_xlabel('K')
            # ax[i,j].label_outer()
            count =+ 1
    # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
    ax[0,0].set_title('Groups = 10')
    ax[0,1].set_title('Groups = 50')
    ax[0,2].set_title('Groups = 100')

    ax[0,0].set_ylabel('Obs per group = 30 \n Predictive Log Likelihood')
    ax[1,0].set_ylabel('Obs per group = 200 \n Predictive Log Likelihood')

    ax[1,0].sharex(ax[0,0])
    ax[1,0].set_xlabel('K')
    ax[1,1].sharex(ax[0,0])
    ax[1,1].set_xlabel('K')
    ax[1,2].sharex(ax[0,0])
    ax[1,2].set_xlabel('K')
    # fig.tight_layout()
    plt.legend()
    plt.savefig('charts/chart_movielens_rws.png'.format(N, M))
    plt.savefig('charts/chart_movielens_rws.pdf'.format(N, M))
