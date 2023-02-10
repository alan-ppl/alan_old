import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from tueplots import axes, bundles, figsizes

#Ks = ['1','3','10','30']
Ks = ['3','10','30']
Ns = ['2']
Ms = ['2']
# with open('results.json') as f:
#     results = json.load(f)
#
# with open('results_local_IW.json') as f:
#     results_local_IW = json.load(f)


plt.rcParams.update(bundles.icml2022())
# plt.rcParams.update(figsizes.icml2022_half())
count = 0
fig, ax = plt.subplots(1,1)
for i in range(len(Ns)):
    for j in range(len(Ms)):
        N = Ns[i]
        M = Ms[j]
        # with open('results/radon_discrete/rws_tmc_new_LIW_N2_M2.json') as f:
        #     results_local_IW = json.load(f)
        #
        with open('results/radon_discrete/rws_tmc_N2_M2.json') as f:
            results_rws_tmc = json.load(f)

        with open('results/radon_discrete/rws_tmc_new_N2_M2.json') as f:
            results_rws_tmc_new = json.load(f)

        with open('results/radon_discrete/rws_global_N2_M2.json') as f:
            results_rws_global_k = json.load(f)


        elbos_rws_tmc = [results_rws_tmc[N][M][k]['pred_likelihood'] for k in Ks]
        stds_rws_tmc = [results_rws_tmc[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]

        elbos_rws_tmc_new = [results_rws_tmc_new[N][M][k]['pred_likelihood'] for k in Ks]
        stds_rws_tmc_new = [results_rws_tmc_new[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]

        elbos_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood'] for k in Ks]
        stds_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]


        ax.errorbar(Ks,elbos_rws_global_k, yerr=stds_rws_global_k, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global RWS')
        ax.errorbar(Ks,elbos_rws_tmc, yerr=stds_rws_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='orange', label='TMC RWS')
        ax.errorbar(Ks,elbos_rws_tmc_new, yerr=stds_rws_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP RWS')

        # ax.set_ylabel('Final Lower Bound')
        # ax.set_xlabel('K')
        # ax[i,j].label_outer()
        count =+ 1
# plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
# ax.set_title('4 States, 4 Counties, 4 Zipcodes, 2 Readings')

ax.set_ylabel('Predictive Log Likelihood')


ax.set_xlabel('K')
plt.legend()
plt.savefig('charts/chart_radon_discrete.png')
plt.savefig('charts/chart_radon_discrete.pdf')
