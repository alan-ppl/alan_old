import numpy as np
import matplotlib.pyplot as plt
import json
from tueplots import axes, bundles

Ks = ['1','5','10','15']
Ns = ['10','30']
Ms = ['10','50','100']
with open('results.json') as f:
    results = json.load(f)


for N in Ns:
    for M in Ms:
        plt.rcParams.update({"figure.dpi": 150})
        with plt.rc_context(bundles.icml2022()):
            fig, ax = plt.subplots(figsize=(6, 6))

            elbos = [results[N][M][k]['lower_bound'] for k in Ks]
            stds = [results[N][M][k]['std'] for k in Ks]

            # lower_error = [elbos[i] - stds[i] for i in range(len(elbos))]
            # upper_error = [elbos[i] + stds[i] for i in range(len(elbos))]

            ax.errorbar(Ks,elbos, yerr=stds, linewidth=0.75, fmt='-o')
            plt.title('Groups: {0}, Observations per group: {1}, with one standard deviation'.format(M, N))
            ax.set_ylabel('Test log likelihood')
            ax.set_xlabel('K')
            plt.savefig('N_{0}_M_{1}.png'.format(N, M))
            plt.savefig('N_{0}_M_{1}.png'.format(N, M))
