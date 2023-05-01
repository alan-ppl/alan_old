import numpy as np
import matplotlib.pyplot as plt
import pickle
from tueplots import axes, bundles
from tueplots import cycler
from tueplots.constants import markers
from tueplots.constants.color import palettes

from alan.experiment_utils import n_mean
Ks = ['3','10','30']

N = '2'
M = '2'
lrs = ['0.003', '0.001', '0.0003', '0.0001']
mean_no = 50

ml_colours = ['#f1eef6','#bdc9e1','#74a9cf','#2b8cbe','#045a8d'][::-1]
adam_colours = ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026'][::-1]
def plot():
    fig_pll, ax_pll = plt.subplots(1,len(Ks), figsize=(5.5, 2.0), sharey=True)
    fig_elbo, ax_elbo = plt.subplots(1,len(Ks), figsize=(5.5, 2.0))
    for K in range(len(Ks)):

        for lr in range(len(lrs)):
            #ML
            try:
                with open('results/bus_breakdown/ML_3500_{}_K{}_True.pkl'.format(lrs[lr],Ks[K]), 'rb') as f:
                    results_ml_tmc_new = pickle.load(f)


                #pred_ll
                elbos_ml_tmc_new = n_mean(results_ml_tmc_new['pred_likelihood'], mean_no).mean(axis=0)
                stds_ml_tmc_new = n_mean(results_ml_tmc_new['pred_likelihood'], mean_no).std(axis=0) / np.sqrt(10)
                time_ml_tmc_new = n_mean(results_ml_tmc_new['times'], mean_no).mean(axis=0).cumsum(axis=0)
                ax_pll[K].errorbar(time_ml_tmc_new,elbos_ml_tmc_new, yerr=stds_ml_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-',color=ml_colours[lr], label='ML lr: {}'.format(lrs[lr], Ks[K]))

                #elbos
                elbos_ml_tmc_new = n_mean(results_ml_tmc_new['objs'], mean_no).mean(axis=0)
                elbos_stds_ml_tmc_new = n_mean(results_ml_tmc_new['objs'], mean_no).std(axis=0) / np.sqrt(10)
                ax_elbo[K].errorbar(time_ml_tmc_new,elbos_ml_tmc_new, yerr=elbos_stds_ml_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-',color=ml_colours[lr], label='ML lr: {}'.format(lrs[lr], Ks[K]))
                del results_ml_tmc_new
            except:
                None
            #VI
            try:
                with open('results/bus_breakdown/VI_3500_{}_K{}_True.pkl'.format(lrs[lr],Ks[K]), 'rb') as f:
                    results_adam_tmc_new = pickle.load(f)
                #Pred_ll
                elbos_adam_tmc_new = n_mean(results_adam_tmc_new['pred_likelihood'], mean_no).mean(axis=0)
                stds_adam_tmc_new = n_mean(results_adam_tmc_new['pred_likelihood'], mean_no).std(axis=0) / np.sqrt(10)
                time_adam_tmc_new = n_mean(results_adam_tmc_new['times'], mean_no).mean(axis=0).cumsum(axis=0)
                ax_pll[K].errorbar(time_adam_tmc_new,elbos_adam_tmc_new, yerr=stds_adam_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-',color=adam_colours[lr], label='MP VI lr: {}'.format(lrs[lr], Ks[K]))
                #Elbo
                elbos_adam_tmc_new = n_mean(results_adam_tmc_new['objs'], mean_no).mean(axis=0)
                elbos_stds_adam_tmc_new = n_mean(results_adam_tmc_new['objs'], mean_no).std(axis=0) / np.sqrt(10)
                ax_elbo[K].errorbar(time_adam_tmc_new,elbos_adam_tmc_new, yerr=elbos_stds_adam_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-',color=adam_colours[lr], label='MP VI lr: {}'.format(lrs[lr]))
                del results_adam_tmc_new
            except:
                None

        if K == 0:
            ax_pll[K].set_ylabel('Predictive Log Likelihood')
            ax_elbo[K].set_ylabel('Elbo')

        if K == 0:
            fig_pll.legend(loc='lower center', bbox_to_anchor=(0.55, -0.15),
                           ncol=5)
            fig_elbo.legend(loc='lower center', bbox_to_anchor=(0.55, -0.15),
                           ncol=5)

        ax_pll[K].set_title('{}'.format('abcd'[K]))
        ax_elbo[K].set_title('{}'.format('abcd'[K]))

        ax_pll[K].set_xlabel('Time (s)')


        ax_elbo[K].set_xlabel('Time (s)')

    fig_pll.savefig('charts/chart_bus_breakdown_ml1_predll.png')
    fig_pll.savefig('charts/chart_bus_breakdown_ml1_predll.pdf')



    fig_elbo.savefig('charts/chart_time_bus_breakdown_ml1_elbo.png')
    fig_elbo.savefig('charts/chart_time_bus_breakdown_ml1_elbo.pdf')

def plot_moments():
    fig_pll, ax_pll = plt.subplots(1,len(Ks), figsize=(5.5, 2.0), sharey=True)
    fig_elbo, ax_elbo = plt.subplots(1,len(Ks), figsize=(5.5, 2.0), sharey=True)
    for K in range(len(Ks)):
        for lr in range(len(lrs)):
            #ML
            try:
                with open('results/bus_breakdown/ML_3500_{}_K{}_False.pkl'.format(lrs[lr],Ks[K]), 'rb') as f:
                    results_ml_tmc_new = pickle.load(f)

                #moments
                elbos_ml_tmc_new = n_mean(results_ml_tmc_new['sq_errs'], mean_no).mean(axis=0)
                stds_ml_tmc_new = n_mean(results_ml_tmc_new['sq_errs'], mean_no).std(axis=0) / np.sqrt(10)
                time_ml_tmc_new = n_mean(results_ml_tmc_new['times'], mean_no).mean(axis=0).cumsum()
                ax_pll[K].errorbar(time_ml_tmc_new,elbos_ml_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-',color=ml_colours[lr], label='ML lr: {}'.format(lrs[lr], Ks[K]))
                del results_ml_tmc_new
            except:
                None
            #VI
            try:
                with open('results/bus_breakdown/VI_3500_{}_K{}_False.pkl'.format(lrs[lr],Ks[K]), 'rb') as f:
                    results_adam_tmc_new = pickle.load(f)
                #moments
                elbos_adam_tmc_new = n_mean(results_adam_tmc_new['sq_errs'], mean_no).mean(axis=0)
                stds_adam_tmc_new = n_mean(results_adam_tmc_new['sq_errs'], mean_no).std(axis=0) / np.sqrt(10)
                time_adam_tmc_new = n_mean(results_adam_tmc_new['times'], mean_no).mean(axis=0).cumsum()
                ax_pll[K].errorbar(time_adam_tmc_new,elbos_adam_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-',color=adam_colours[lr], label='MP VI lr: {}'.format(lrs[lr], Ks[K]))
                del results_adam_tmc_new
            except:
                None
        if K == 0:
            ax_pll[K].set_ylabel('Average Moment MSE')
            ax_elbo[K].set_ylabel('Elbo')

        if K == 0:
            fig_pll.legend(loc='lower center', bbox_to_anchor=(0.55, -0.15),
                           ncol=5)
            fig_elbo.legend(loc='lower center', bbox_to_anchor=(0.55, -0.15),
                           ncol=5)

        ax_pll[K].set_title('{}'.format('abcd'[K]))
        ax_elbo[K].set_title('{}'.format('abcd'[K]))

        ax_pll[K].set_xlabel('Time (s)')


        ax_elbo[K].set_xlabel('Time (s)')
    fig_pll.savefig('charts/chart_bus_breakdown_ml1_moments.png')
    fig_pll.savefig('charts/chart_bus_breakdown_ml1_moments.pdf')


def plot_bars():
    fig_bar, ax_bar = plt.subplots(1,len(Ks), figsize=(5.5, 2.0), sharey=True)
    for K in range(len(Ks)):


        x = np.arange(len(lrs))  # the label locations
        width = 0.3  # the width of the bars
        multiplier = 0

        pred_liks_ml = []
        pred_liks_ml_std = []
        pred_liks_adam = []
        pred_liks_adam_std = []
        for lr in range(len(lrs)):
            #ML
            try:
                with open('results/bus_breakdown/ML_3500_{}_K{}_True.pkl'.format(lrs[lr],Ks[K]), 'rb') as f:
                    results_ml_tmc_new = pickle.load(f)

                #moments
                final_pred_lik_ml = results_ml_tmc_new['final_pred_lik_K=30'].mean(axis=0)
                final_pred_lik_std_err_ml = results_ml_tmc_new['final_pred_lik_K=30'].std(axis=0) / np.sqrt(10)

                pred_liks_ml.append(final_pred_lik_ml)
                pred_liks_ml_std.append(final_pred_lik_std_err_ml)
                del results_ml_tmc_new
            except:
                pred_liks_ml.append(0)
                pred_liks_ml_std.append(0)
            #VI
            try:
                with open('results/bus_breakdown/VI_3500_{}_K{}_True.pkl'.format(lrs[lr],Ks[K]), 'rb') as f:
                    results_adam_tmc_new = pickle.load(f)
                #moments
                final_pred_lik_adam = results_adam_tmc_new['final_pred_lik_K=30'].mean(axis=0)
                final_pred_lik_std_err_adam = results_adam_tmc_new['final_pred_lik_K=30'].std(axis=0) / np.sqrt(10)

                pred_liks_adam.append(final_pred_lik_adam)
                pred_liks_adam_std.append(final_pred_lik_std_err_adam)
                del results_adam_tmc_new
            except:
                pred_liks_adam.append(0)
                pred_liks_adam_std.append(0)



        offset = width * multiplier
        if K>0:
            rects = ax_bar[K].bar(x + offset, pred_liks_ml, width, yerr=pred_liks_ml_std, color=ml_colours[0])
            # ax_bar.bar_label(rects, padding=3)
            multiplier += 1

            offset = width * multiplier
            rects = ax_bar[K].bar(x + offset, pred_liks_adam, width, yerr=pred_liks_adam_std, color=adam_colours[0])
            # ax_bar.bar_label(rects, padding=3)
            multiplier += 1
        else:
            rects = ax_bar[K].bar(x + offset, pred_liks_ml, width, yerr=pred_liks_ml_std,color=ml_colours[0], label='ML')
            # ax_bar.bar_label(rects, padding=3)
            multiplier += 1

            offset = width * multiplier
            rects = ax_bar[K].bar(x + offset, pred_liks_adam, width, yerr=pred_liks_adam_std, color=adam_colours[0], label='MP VI')
            # ax_bar.bar_label(rects, padding=3)
            multiplier += 1
        ax_bar[K].set_xticks(x + width, lrs)
        ax_bar[K].set_xlabel('Learning rate')
        if K == 0:
            ax_bar[K].set_ylabel(r'Predictive log likelihood' + '\n' +'evaluated for K=30')

        ax_bar[K].set_title('{}'.format('abcd'[K]))
        # ax_bar[K].set_ylim(-1075,-920)
    # ax_pll.set_xlim(-1,20)

    fig_bar.legend(loc='lower center', bbox_to_anchor=(0.55, -0.15),
                   ncol=5)
    fig_bar.savefig('charts/chart_bus_breakdown_bar.png')
    fig_bar.savefig('charts/chart_bus_breakdown_bar.pdf')

plt.rcParams.update({"figure.dpi": 3500})
# plt.rcParams.update(cycler.cycler(color=palettes.muted))
with plt.rc_context(bundles.icml2022()):

    plot()
    plot_moments()
    plot_bars()
