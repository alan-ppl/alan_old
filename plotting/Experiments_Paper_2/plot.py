import numpy as np
import matplotlib.pyplot as plt
import json
from tueplots import axes, bundles

Ks = ['1','5','10','15']
Ns = ['10','30']
Ms = ['10','50','100']
with open('results.json') as f:
    results = json.load(f)

with open('results_local_IW.json') as f:
    results_local_IW = json.load(f)

plt.rcParams.update({"figure.dpi": 150})
# with plt.rc_context(bundles.icml2022()):


for N in Ns:
    for M in Ms:
        fig, ax = plt.subplots(figsize=(6, 6))

        print(N)
        print(M)
        elbos = [results[N][M][k]['lower_bound'] for k in Ks]
        stds = [results[N][M][k]['std'] for k in Ks]

        elbos_IW = [results_local_IW[N][M][k]['lower_bound'] for k in Ks[1:]]
        stds_IW = [results_local_IW[N][M][k]['std'] for k in Ks[1:]]


        ax.errorbar(Ks,[elbos[0]] + elbos_IW, yerr=[stds[0]] + stds_IW, linewidth=0.75, fmt='-o', c='red', label='LIW')
        ax.errorbar(Ks,elbos, yerr=stds, linewidth=0.75, fmt='-o', c='blue', label='TPP')
        plt.title('Groups: {0}, Observations per group: {1}, with one standard deviation'.format(M, N))
        ax.set_ylabel('Final Lower Bound')
        ax.set_xlabel('K')
        ax.legend()
        plt.savefig('N_{0}_M_{1}.png'.format(N, M))
        plt.savefig('N_{0}_M_{1}.pdf'.format(N, M))
