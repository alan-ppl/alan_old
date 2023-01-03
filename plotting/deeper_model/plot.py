import numpy as np
import matplotlib.pyplot as plt
import json
from tueplots import axes, bundles

Ks = ['1','5','10','15']
Ns = ['2']
Ms = ['2','4','10']
# with open('results.json') as f:
#     results = json.load(f)
#
# with open('results_local_IW.json') as f:
#     results_local_IW = json.load(f)

plt.rcParams.update({"figure.dpi": 300})
with plt.rc_context(bundles.icml2022()):

    count = 0
    fig, ax = plt.subplots(1,3, figsize=(5.5, 3.5))

    for i in range(len(Ns)):
        for j in range(len(Ms)):
            N = Ns[i]
            M = Ms[j]
            with open('results/results_LIW_N{0}_M{1}.json'.format(N,M)) as f:
                results_local_IW = json.load(f)

            with open('results/results_N{0}_M{1}.json'.format(N,M)) as f:
                results = json.load(f)

            with open('results/results_global_K_N{0}_M{1}.json'.format(N,M)) as f:
                results_global_K = json.load(f)

            with open('results/results_tmc_N{0}_M{1}.json'.format(N,M)) as f:
                results_tmc = json.load(f)

            elbos_tpp = [results[N][M][k]['lower_bound'] for k in Ks]
            stds_tpp = [results[N][M][k]['std']/np.sqrt(5) for k in Ks]

            elbos_IW = [results_local_IW[N][M][k]['lower_bound'] for k in Ks]
            stds_IW = [results_local_IW[N][M][k]['std']/np.sqrt(5) for k in Ks]

            elbos_global_K = [results_global_K[N][M][k]['lower_bound'] for k in Ks]
            stds_global_K = [results_global_K[N][M][k]['std']/np.sqrt(5) for k in Ks]

            elbos_tmc = [results_tmc[N][M][k]['lower_bound'] for k in Ks]
            stds_tmc = [results_tmc[N][M][k]['std']/np.sqrt(5) for k in Ks]


            ax[j].errorbar(Ks,elbos_IW, yerr=stds_IW, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='LIW')
            ax[j].errorbar(Ks,elbos_tpp, yerr=stds_tpp, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='TPP')
            ax[j].errorbar(Ks,elbos_global_K, yerr=stds_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='Global K')
            ax[j].errorbar(Ks,elbos_tmc, yerr=stds_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='orange', label='TMC')
            #
            # ax.set_ylabel('Final Lower Bound')
            # ax.set_xlabel('K')
            # ax[i,j].label_outer()
            count =+ 1
    # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
    ax[0].set_title('Groups = 2')
    ax[1].set_title('Groups = 4')
    ax[2].set_title('Groups = 10')

    ax[0].set_ylabel('Obs per group = 2 \n Final Lower Bound')

    # ax[1,0].sharex(ax[0,0])
    ax[0].set_xlabel('K')
    # ax[1,1].sharex(ax[0,0])
    ax[1].set_xlabel('K')
    # ax[1,2].sharex(ax[0,0])
    ax[2].set_xlabel('K')
    # fig.tight_layout()
    plt.legend()
    plt.savefig('charts/chart_deeper_regression.png'.format(N, M))
    plt.savefig('charts/chart_deeper_regression.pdf'.format(N, M))

    # count = 0
    # fig, ax = plt.subplots(1,3, figsize=(5.5, 3.5))
    #
    # for i in range(len(Ns)):
    #     for j in range(len(Ms)):
    #         N = Ns[i]
    #         M = Ms[j]
    #         with open('results/results_LIW_N{0}_M{1}.json'.format(N,M)) as f:
    #             results_local_IW = json.load(f)
    #
    #         with open('results/results_N{0}_M{1}.json'.format(N,M)) as f:
    #             results_tpp = json.load(f)
    #
    #
    #         log_likes_IW = [results_local_IW[N][M][k]['test_log_likelihood_mean'] for k in Ks]
    #         std_log_likes_IW = [results_local_IW[N][M][k]['test_log_likelihood_std']/np.sqrt(5) for k in Ks]
    #
    #         log_likes = [results_tpp[N][M][k]['test_log_likelihood_mean'] for k in Ks]
    #         std_log_likes = [results_tpp[N][M][k]['test_log_likelihood_std']/np.sqrt(5) for k in Ks]
    #
    #         ax[j].errorbar(Ks,log_likes_IW, yerr=std_log_likes_IW, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='LIW')
    #         ax[j].errorbar(Ks,log_likes, yerr=std_log_likes, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='TPP')
    #         #
    #         # ax.set_ylabel('Final Lower Bound')
    #         # ax.set_xlabel('K')
    #         # ax[i,j].label_outer()
    #         count =+ 1
    #
    # ax[0].set_title('Groups = 2')
    # ax[1].set_title('Groups = 4')
    # ax[2].set_title('Groups = 10')
    #
    # ax[0].set_ylabel('Obs per group = 2 \n Test log-likelihood')
    #
    # # ax[1,0].sharex(ax[0,0])
    # ax[0].set_xlabel('K')
    # # ax[1,1].sharex(ax[0,0])
    # ax[1].set_xlabel('K')
    # # ax[1,2].sharex(ax[0,0])
    # ax[2].set_xlabel('K')
    # # fig.tight_layout()
    # plt.legend()
    # plt.savefig('charts/chart_test_deeper_regression.png'.format(N, M))
    # plt.savefig('charts/chart_test_deeper_regression.pdf'.format(N, M))
