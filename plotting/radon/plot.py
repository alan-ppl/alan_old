import numpy as np
import matplotlib.pyplot as plt
import json
from tueplots import axes, bundles

Ks = ['1','3','10','30']
Ns = ['2']
Ms = ['2']
# with open('results.json') as f:
#     results = json.load(f)
#
# with open('results_local_IW.json') as f:
#     results_local_IW = json.load(f)

plt.rcParams.update({"figure.dpi": 300})
with plt.rc_context(bundles.icml2022()):

    count = 0
    fig, ax = plt.subplots(1,1, figsize=(5.5, 3.5))
    for i in range(len(Ns)):
        for j in range(len(Ms)):
            N = Ns[i]
            M = Ms[j]
            with open('results/elbo_LIW_N2_M2.json') as f:
                results_local_IW = json.load(f)

            with open('results/elbo_N2_M2.json') as f:
                results = json.load(f)

            with open('results/elbo_global_N2_M2.json') as f:
                results_global_K = json.load(f)


            elbos_tpp = [results[N][M][k]['final_obj'] for k in Ks]
            stds_tpp = [results[N][M][k]['final_obj_std']/np.sqrt(5) for k in Ks]

            elbos_IW = [results_local_IW[N][M][k]['final_obj'] for k in Ks]
            stds_IW = [results_local_IW[N][M][k]['final_obj_std']/np.sqrt(5) for k in Ks]

            elbos_global_K = [results_global_K[N][M][k]['final_obj'] for k in Ks]
            stds_global_K = [results_global_K[N][M][k]['final_obj_std']/np.sqrt(5) for k in Ks]


            ax.errorbar(Ks,elbos_IW, yerr=stds_IW, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='LIW')
            ax.errorbar(Ks,elbos_tpp, yerr=stds_tpp, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='TPP')
            ax.errorbar(Ks,elbos_global_K, yerr=stds_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='Global K')
            #
            # ax.set_ylabel('Final Lower Bound')
            # ax.set_xlabel('K')
            # ax[i,j].label_outer()
            count =+ 1
    # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
    ax.set_title('2 States, 2 Counties, 4 Zipcodes, 4 Readings')

    ax.set_ylabel('Final Lower Bound')


    ax.set_xlabel('K')
    plt.legend()
    plt.savefig('charts/chart_radon_unif.png')
    plt.savefig('charts/chart_radon_unif.pdf')
