import argparse
import numpy as np
import matplotlib.pyplot as plt

from tueplots import axes, bundles

parser = argparse.ArgumentParser(description='Plotting')

parser.add_argument('N', type=int,
                    help='Scale of experiment')

args = parser.parse_args()
N = args.N

# Increase the resolution of all the plots below
for type in ['Dense', 'Diagonal', 'Block']:
    plt.rcParams.update({"figure.dpi": 150})
    with plt.rc_context(bundles.icml2022()):
        fig, ax = plt.subplots(figsize=(6, 6))
        if N == 10:
            log_prob = np.load('log_prob.npy')
            ax.plot(range(200000),[log_prob]*200000, label="Log Marg", color='black', linewidth=1)

        for K in [1, 5, 10, 20, 50, 100]:
            x = np.load('{0}_K{1}_N{2}.npy'.format(type, K, N))
            ### plotting elbos
            def ewm(data, alpha):
                out = []
                for i in range(data.shape[0]):
                    if i == 0:
                        out.append(data[i])
                    else:
                        out.append(alpha* data[i] + (1-alpha)*out[i-1])
                return np.array(out)


            y = ewm(x, 0.001)
            # y = x

            ax.plot(range(y.shape[0]),y, label="K={}".format(K), linewidth=0.75)


    plt.ylim([log_prob-10, log_prob+1])
    plt.xlim([0, 200000])
    ax.legend()
    plt.savefig('{0}_N{1}.png'.format(type, N))
    plt.savefig('{0}_N{1}.pdf'.format(type, N))
