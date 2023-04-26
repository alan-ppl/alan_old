import numpy as np
import matplotlib.pyplot as plt
import json
from tueplots import axes, bundles
from tueplots import cycler
from tueplots.constants import markers
from tueplots.constants.color import palettes

Ks = ['3','10','30']

N = '2'
M = '2'
lrs = ['0.01', '0.003', '0.001', '0.0003', '0.0001']

def plot():
    fig_pll, ax_pll = plt.subplots(1,len(Ks), figsize=(5.5, 2.0), sharey=True)
    fig_elbo, ax_elbo = plt.subplots(1,len(Ks), figsize=(5.5, 2.0))
    for K in range(len(Ks)):

        for lr in range(len(lrs)):
            #ML
            try:
                with open('results/bus_breakdown/ML_3500_{}_N{}_M{}_K{}_True.json'.format(lrs[lr],N,M,Ks[K])) as f:
                    results_ml_tmc_new = json.load(f)

                #pred_ll
                elbos_ml_tmc_new = results_ml_tmc_new[N][M][Ks[K]]['pred_likelihood']
                stds_ml_tmc_new = results_ml_tmc_new[N][M][Ks[K]]['pred_likelihood_std']
                time_ml_tmc_new = results_ml_tmc_new[N][M][Ks[K]]['time']
                ax_pll[K].errorbar(time_ml_tmc_new,elbos_ml_tmc_new, yerr=stds_ml_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o',color='r', alpha=1/(lr+1), label='ML lr: {}'.format(lrs[lr], Ks[K]))

                #elbos
                elbos_ml_tmc_new = results_ml_tmc_new[N][M][Ks[K]]['objs']
                elbos_stds_ml_tmc_new = results_ml_tmc_new[N][M][Ks[K]]['obj_stds']

                ax_elbo[K].errorbar(time_ml_tmc_new,elbos_ml_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o',color='r', alpha=1/(lr+1), label='ML lr: {}'.format(lrs[lr], Ks[K]))

            except:
                None
            #VI
            try:
                with open('results/bus_breakdown/VI_3500_{}_N{}_M{}_K{}_True.json'.format(lrs[lr],N,M,Ks[K])) as f:
                    results_adam_tmc_new = json.load(f)
                #Pred_ll
                elbos_adam_tmc_new = results_adam_tmc_new[N][M][Ks[K]]['pred_likelihood']
                stds_adam_tmc_new = results_adam_tmc_new[N][M][Ks[K]]['pred_likelihood_std']
                time_adam_tmc_new = results_adam_tmc_new[N][M][Ks[K]]['time']
                ax_pll[K].errorbar(time_adam_tmc_new,elbos_adam_tmc_new,yerr=stds_adam_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o',color='c', alpha=1/(lr+1), label='MP VI lr: {}'.format(lrs[lr], Ks[K]))
                #Elbo
                elbos_adam_tmc_new = results_adam_tmc_new[N][M][Ks[K]]['objs']
                elbos_stds_adam_tmc_new = results_adam_tmc_new[N][M][Ks[K]]['obj_stds']
                ax_elbo[K].errorbar(time_adam_tmc_new,elbos_adam_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o',color='c', alpha=1/(lr+1), label='MP VI lr: {}'.format(lrs[lr]))

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
        # ax_elbo[K].set_ylim(-5300,-1900)
        ax_pll[K].set_xlabel('Time (s)')


        ax_elbo[K].set_xlabel('Time (s)')

    fig_pll.savefig('charts/chart_bus_breakdown_ml1_predll_N{}_M{}.png'.format(N, M, Ks[K]))
    fig_pll.savefig('charts/chart_bus_breakdown_ml1_predll_N{}_M{}.pdf'.format(N, M, Ks[K]))



    fig_elbo.savefig('charts/chart_time_bus_breakdown_ml1_elbo_N{}_M{}.png'.format(N, M, Ks[K]))
    fig_elbo.savefig('charts/chart_time_bus_breakdown_ml1_elbo_N{}_M{}.pdf'.format(N, M, Ks[K]))

def plot_moments():
    fig_pll, ax_pll = plt.subplots(1,len(Ks), figsize=(5.5, 2.0), sharey=True)
    fig_elbo, ax_elbo = plt.subplots(1,len(Ks), figsize=(5.5, 2.0), sharey=True)
    for K in range(len(Ks)):
        for lr in range(len(lrs)):
            #ML
            try:
                with open('results/bus_breakdown/ML_3500_{}_N{}_M{}_K{}_False.json'.format(lrs[lr],N,M,Ks[K])) as f:
                    results_ml_tmc_new = json.load(f)

                #moments
                elbos_ml_tmc_new = results_ml_tmc_new[N][M][Ks[K]]['sq_errs']
                stds_ml_tmc_new = results_ml_tmc_new[N][M][Ks[K]]['sq_errs_std']
                time_ml_tmc_new = results_ml_tmc_new[N][M][Ks[K]]['time']
                if not lrs[lr] == '0.1':
                    ax_pll[K].errorbar(time_ml_tmc_new,elbos_ml_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o',color='r', alpha=1/(lr+1), label='ML lr: {}'.format(lrs[lr], Ks[K]))
                else:
                    ax_pll[K].errorbar(time_ml_tmc_new + [np.nan]*250,elbos_ml_tmc_new+ [np.nan]*250, linewidth=0.55, markersize = 0.75, fmt='-o',color='r', alpha=1/(lr+1), label='ML lr: {}'.format(lrs[lr], Ks[K]))
            except:
                None
            #VI
            try:
                with open('results/bus_breakdown/VI_3500_{}_N{}_M{}_K{}_False.json'.format(lrs[lr],N,M,Ks[K])) as f:
                    results_adam_tmc_new = json.load(f)
                #moments
                elbos_adam_tmc_new = results_adam_tmc_new[N][M][Ks[K]]['sq_errs']
                stds_adam_tmc_new = results_adam_tmc_new[N][M][Ks[K]]['sq_errs_std']
                time_adam_tmc_new = results_adam_tmc_new[N][M][Ks[K]]['time']
                ax_pll[K].errorbar(time_adam_tmc_new,elbos_adam_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o',color='c', alpha=1/(lr+1), label='MP VI lr: {}'.format(lrs[lr], Ks[K]))

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
    fig_pll.savefig('charts/chart_bus_breakdown_ml1_moments_N{}_M{}.png'.format(N, M))
    fig_pll.savefig('charts/chart_bus_breakdown_ml1_moments_N{}_M{}.pdf'.format(N, M))


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
                with open('results/bus_breakdown/ML_3500_{}_N{}_M{}_K{}_True.json'.format(lrs[lr],N,M,Ks[K])) as f:
                    results_ml_tmc_new = json.load(f)

                #moments
                final_pred_lik_ml = results_ml_tmc_new[N][M][Ks[K]]['final_pred_lik_K=30']
                final_pred_lik_std_err_ml = results_ml_tmc_new[N][M][Ks[K]]['final_pred_lik_K=30_stderr']

                pred_liks_ml.append(final_pred_lik_ml)
                pred_liks_ml_std.append(final_pred_lik_std_err_ml)
            except:
                pred_liks_ml.append(0)
                pred_liks_ml_std.append(0)
            #VI
            try:
                with open('results/bus_breakdown/VI_3500_{}_N{}_M{}_K{}_True.json'.format(lrs[lr],N,M,Ks[K])) as f:
                    results_adam_tmc_new = json.load(f)
                #moments
                final_pred_lik_adam = results_adam_tmc_new[N][M][Ks[K]]['final_pred_lik_K=30']
                final_pred_lik_std_err_adam = results_adam_tmc_new[N][M][Ks[K]]['final_pred_lik_K=30_stderr']

                pred_liks_adam.append(final_pred_lik_adam)
                pred_liks_adam_std.append(final_pred_lik_std_err_adam)

            except:
                pred_liks_adam.append(0)
                pred_liks_adam_std.append(0)



        offset = width * multiplier
        if K>0:
            rects = ax_bar[K].bar(x + offset, pred_liks_ml, width, yerr=pred_liks_ml_std)
            # ax_bar.bar_label(rects, padding=3)
            multiplier += 1

            offset = width * multiplier
            rects = ax_bar[K].bar(x + offset, pred_liks_adam, width, yerr=pred_liks_adam_std)
            # ax_bar.bar_label(rects, padding=3)
            multiplier += 1
        else:
            rects = ax_bar[K].bar(x + offset, pred_liks_ml, width, yerr=pred_liks_ml_std, label='ML')
            # ax_bar.bar_label(rects, padding=3)
            multiplier += 1

            offset = width * multiplier
            rects = ax_bar[K].bar(x + offset, pred_liks_adam, width, yerr=pred_liks_adam_std, label='MP VI')
            # ax_bar.bar_label(rects, padding=3)
            multiplier += 1
        ax_bar[K].set_xticks(x + width, lrs)
        ax_bar[K].set_xlabel('Learning rate')
        if K == 0:
            ax_bar[K].set_ylabel(r'Predictive log likelihood' + '\n' +'evaluated for K=30')

        ax_bar[K].set_title('{}'.format('abcd'[K]))
    # ax_pll.set_ylim(-1000,-900)
    # ax_pll.set_xlim(-1,20)

    fig_bar.legend(loc='lower center', bbox_to_anchor=(0.55, -0.15),
                   ncol=5)
    fig_bar.savefig('charts/chart_bus_breakdown_bar_N{}_M{}_K{}.png'.format(N, M, Ks[K]))
    fig_bar.savefig('charts/chart_bus_breakdown_bar_N{}_M{}_K{}.pdf'.format(N, M, Ks[K]))

plt.rcParams.update({"figure.dpi": 1000})
# plt.rcParams.update(cycler.cycler(color=palettes.muted))
with plt.rc_context(bundles.icml2022()):

    plot()
    plot_moments()
    plot_bars()
