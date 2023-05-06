import numpy as np
import matplotlib.pyplot as plt
import pickle
from tueplots import axes, bundles
from tueplots import cycler
from tueplots.constants import markers
from tueplots.constants.color import palettes
import matplotlib.patches as mpatches
from alan.experiment_utils import n_mean
Ks = ['3','10','30']

N = '5'
M = '300'
lrs = ['0.1', '0.03', '0.01']# '0.003', '0.001']
mean_no = 10

ml_colours = ['#f1eef6','#bdc9e1','#74a9cf','#2b8cbe','#045a8d'][::-1]
adam_colours = ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026'][::-1]

plt.rcParams.update({"figure.dpi": 1000})
# plt.rcParams.update(cycler.cycler(color=palettes.muted))
with plt.rc_context(bundles.icml2022()):
    for data in ['True', 'False']:
        fig, ax = plt.subplots(3,len(Ks), figsize=(5.5, 8.0), constrained_layout=True)
        for K in range(len(Ks)):

            for lr in range(len(lrs)):
                #ML
                try:
                    with open('results/movielens/ML_{}_{}_K{}_{}.pkl'.format(750 if lrs[lr]=='0.1' else 1000, lrs[lr],Ks[K], data), 'rb') as f:
                        results_ml_tmc_new = pickle.load(f)


                    #pred_ll
                    elbos_ml_tmc_new = n_mean(results_ml_tmc_new['pred_likelihood'], mean_no).mean(axis=0)
                    stds_ml_tmc_new = n_mean(results_ml_tmc_new['pred_likelihood'], mean_no).std(axis=0) / np.sqrt(10)
                    time_ml_tmc_new = results_ml_tmc_new['times'].mean(axis=0).cumsum(axis=0)[::mean_no]
                    time_ml_tmc_new[time_ml_tmc_new > 12] = np.nan
                    ax[1,K].errorbar(time_ml_tmc_new,elbos_ml_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-',color=ml_colours[lr])

                    #elbos
                    elbos_ml_tmc_new = n_mean(results_ml_tmc_new['objs'], mean_no).mean(axis=0)
                    elbos_stds_ml_tmc_new = n_mean(results_ml_tmc_new['objs'], mean_no).std(axis=0) / np.sqrt(10)
                    if not lrs[lr] == '0.1':
                        ax[0,K].errorbar(time_ml_tmc_new,elbos_ml_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-',color=ml_colours[lr], label='ML lr: {}'.format(lrs[lr], Ks[K]) if K==0 else None)
                    else:
                        ax[0,K].errorbar(time_ml_tmc_new.tolist() + [np.nan]*(250//mean_no),elbos_ml_tmc_new.tolist() + [np.nan]*(250//mean_no), linewidth=0.55, markersize = 0.75, fmt='-',color=ml_colours[lr], label='ML lr: {}'.format(lrs[lr], Ks[K]) if K==0 else None)

                except:
                    None
                #VI
                try:
                    with open('results/movielens/VI_1000_{}_K{}_{}.pkl'.format(lrs[lr],Ks[K],data), 'rb') as f:
                        results_adam_tmc_new = pickle.load(f)
                    #Pred_ll
                    elbos_adam_tmc_new = n_mean(results_adam_tmc_new['pred_likelihood'], mean_no).mean(axis=0)
                    stds_adam_tmc_new = n_mean(results_adam_tmc_new['pred_likelihood'], mean_no).std(axis=0) / np.sqrt(10)
                    time_adam_tmc_new = results_adam_tmc_new['times'].mean(axis=0).cumsum(axis=0)[::mean_no]
                    time_adam_tmc_new[time_adam_tmc_new > 12] = np.nan
                    ax[1,K].errorbar(time_adam_tmc_new,elbos_adam_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-',color=adam_colours[lr], label='MP VI lr: {}'.format(lrs[lr], Ks[K]) if K==0 else None)
                    #Elbo
                    elbos_adam_tmc_new = n_mean(results_adam_tmc_new['objs'], mean_no).mean(axis=0)
                    elbos_stds_adam_tmc_new = n_mean(results_adam_tmc_new['objs'], mean_no).std(axis=0) / np.sqrt(10)
                    ax[0,K].errorbar(time_adam_tmc_new,elbos_adam_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-',color=adam_colours[lr])

                except:
                    None


            ax[1,0].set_ylabel('Predictive Log Likelihood')
            ax[0,0].set_ylabel('Elbo')


            # ax[1,2].legend(loc='right', bbox_to_anchor=(2, 0.5),
            #                ncol=2)

            ax[0,K].set_title(f'Training K: {Ks[K]}')

        # handles, labels = fig.get_legend_handles_labels()
        # unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]



        for K in range(len(Ks)):
            for lr in range(len(lrs)):
                #ML
                try:
                    with open('results/movielens/ML_{}_{}_K{}_{}.pkl'.format(750 if lrs[lr]=='0.1' else 1000, lrs[lr],Ks[K], data), 'rb') as f:
                        results_ml_tmc_new = pickle.load(f)

                    #moments
                    elbos_ml_tmc_new = n_mean(results_ml_tmc_new['sq_errs'], mean_no).mean(axis=0)
                    stds_ml_tmc_new = n_mean(results_ml_tmc_new['sq_errs'], mean_no).std(axis=0) / np.sqrt(10)
                    time_ml_tmc_new = results_ml_tmc_new['times'].mean(axis=0).cumsum(axis=0)[::mean_no]
                    time_ml_tmc_new[time_ml_tmc_new > 12] = np.nan
                    if not lrs[lr] == '0.1':
                        ax[2,K].errorbar(time_ml_tmc_new,elbos_ml_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-',color=ml_colours[lr])
                    else:
                        ax[2,K].errorbar(time_ml_tmc_new + [np.nan]*250/mean_no,elbos_ml_tmc_new+ [np.nan]*250/mean_no, linewidth=0.55, markersize = 0.75, fmt='-',color=ml_colours[lr])
                except:
                    None
                #VI
                try:
                    with open('results/movielens/VI_1000_{}_K{}_{}.pkl'.format(lrs[lr],Ks[K],data), 'rb') as f:
                        results_adam_tmc_new = pickle.load(f)
                    #moments
                    elbos_adam_tmc_new = n_mean(results_adam_tmc_new['sq_errs'], mean_no).mean(axis=0)
                    stds_adam_tmc_new = n_mean(results_adam_tmc_new['sq_errs'], mean_no).std(axis=0) / np.sqrt(10)
                    time_adam_tmc_new = results_adam_tmc_new['times'].mean(axis=0).cumsum(axis=0)[::mean_no]
                    time_adam_tmc_new[time_adam_tmc_new > 12] = np.nan
                    ax[2,K].errorbar(time_adam_tmc_new,elbos_adam_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-',color=adam_colours[lr])

                except:
                    None
            if data == 'True':
                ax[2,0].set_ylabel('Variance for latent: ``z"')
            else:
                ax[2,0].set_ylabel('Average latent MSE')


            # if K == 0:
            #     ax[2,0].legend(loc='center left', bbox_to_anchor=(0.55, -0.15),
            #                    ncol=2)


            ax[2,K].set_xlabel('Time (s)')

        fig.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.55, -0.025))
        # ax[2,1].legend(loc='lower center', bbox_to_anchor=(0.55, -0.15),
        #                ncol=5)
        fig.savefig(f'charts/chart_movielens_{data}.png')
        fig.savefig(f'charts/chart_movielens_{data}.pdf')


    # fig, ax = plt.subplots(len(Ks),len(Ks), figsize=(5.5, 8.0), constrained_layout=True, sharey=True)
    #
    # for K in range(len(Ks)):
    #     for K_2 in range(len(Ks)):
    #
    #         width = 0.3  # the width of the bars
    #         multiplier = 0
    #
    #         pred_liks_ml = []
    #         pred_liks_ml_std = []
    #         pred_liks_adam = []
    #         pred_liks_adam_std = []
    #         for lr in range(len(lrs)):
    #
    #             #ML
    #             try:
    #                 with open('results/movielens/ML_{}_{}_K{}_True.pkl'.format(750 if lrs[lr]=='0.1' else 1000, lrs[lr],Ks[K]), 'rb') as f:
    #                     results_ml_tmc_new = pickle.load(f)
    #
    #                 #moments
    #                 final_pred_lik_ml = results_ml_tmc_new['final_pred_lik_for_K'].mean(axis=0)
    #                 final_pred_lik_std_err_ml = results_ml_tmc_new['final_pred_lik_for_K'].std(axis=0) / np.sqrt(10)
    #
    #                 pred_liks_ml.append(final_pred_lik_ml[K_2])
    #                 pred_liks_ml_std.append(final_pred_lik_std_err_ml[K_2])
    #             except:
    #                 pred_liks_ml.append(0)
    #                 pred_liks_ml_std.append(0)
    #             #VI
    #             try:
    #                 with open('results/movielens/VI_1000_{}_K{}_True.pkl'.format(lrs[lr],Ks[K]), 'rb') as f:
    #                     results_adam_tmc_new = pickle.load(f)
    #                 #moments
    #                 final_pred_lik_adam = results_adam_tmc_new['final_pred_lik_for_K'].mean(axis=0)
    #                 final_pred_lik_std_err_adam = results_adam_tmc_new['final_pred_lik_for_K'].std(axis=0) / np.sqrt(10)
    #
    #                 pred_liks_adam.append(final_pred_lik_adam[K])
    #                 pred_liks_adam_std.append(final_pred_lik_std_err_adam[K])
    #
    #             except:
    #                 pred_liks_adam.append(0)
    #                 pred_liks_adam_std.append(0)
    #
    #
    #
    #         offset = width * multiplier
    #         if K>0 and K_2>0:
    #             rects = ax[K,K_2].bar(x + offset, pred_liks_ml, width, yerr=pred_liks_ml_std, color=ml_colours[0])
    #             # ax_bar.bar_label(rects, padding=3)
    #             multiplier += 1
    #
    #             offset = width * multiplier
    #             rects = ax[K,K_2].bar(x + offset, pred_liks_adam, width, yerr=pred_liks_adam_std, color=adam_colours[0])
    #             # ax_bar.bar_label(rects, padding=3)
    #             multiplier += 1
    #         else:
    #             rects = ax[K,K_2].bar(x + offset, pred_liks_ml, width, yerr=pred_liks_ml_std,color=ml_colours[0], label='ML')
    #             # ax_bar.bar_label(rects, padding=3)
    #             multiplier += 1
    #
    #             offset = width * multiplier
    #             rects = ax[K,K_2].bar(x + offset, pred_liks_adam, width, yerr=pred_liks_adam_std, color=adam_colours[0], label='MP VI')
    #             # ax_bar.bar_label(rects, padding=3)
    #             multiplier += 1
    #
    #         ax[K,K_2].set_xticks(x + width, lrs)
    #         ax[2,1].set_xlabel('Evaluation K')
    #
    #         ax[1,0].set_ylabel(r'Training K')
    #
    #         ax[K,K_2].set_title(f'Training K: {Ks[K]}, Evaluation K: {Ks[K_2]}')
    #
    #
    #
    # blue_patch = mpatches.Patch(color=ml_colours[0], label='ML')
    # red_patch = mpatches.Patch(color=adam_colours[0], label='MP VI')
    # fig.legend(handles = [blue_patch, red_patch], loc='lower center', ncol=6, bbox_to_anchor=(0.55, -0.02))
    # # ax[3,K].legend(loc='lower center', bbox_to_anchor=(0.55, -0.15),
    # #                ncol=5)
    # fig.savefig('charts/chart_movielens_Kplot.png')
    # fig.savefig('charts/chart_movielens_Kplot.pdf')
    #
    #
    # fig, ax = plt.subplots(len(Ks),len(Ks), figsize=(5.5, 8.5), sharey=True)
    #
    # for K in range(len(Ks)):
    #     for K_2 in range(len(Ks)):
    #
    #         width = 0.3  # the width of the bars
    #         multiplier = 0
    #
    #         pred_liks_ml = []
    #         pred_liks_ml_std = []
    #         pred_liks_adam = []
    #         pred_liks_adam_std = []
    #         for lr in range(len(lrs)):
    #
    #             #ML
    #             try:
    #                 with open('results/movielens/ML_1000_{}_K{}_False.pkl'.format(lrs[lr],Ks[K]), 'rb') as f:
    #                     results_ml_tmc_new = pickle.load(f)
    #
    #                 #moments
    #                 final_pred_lik_ml = results_ml_tmc_new['final_pred_lik_for_K'].mean(axis=0)
    #                 final_pred_lik_std_err_ml = results_ml_tmc_new['final_pred_lik_for_K'].std(axis=0) / np.sqrt(10)
    #
    #                 pred_liks_ml.append(final_pred_lik_ml[K_2])
    #                 pred_liks_ml_std.append(final_pred_lik_std_err_ml[K_2])
    #             except:
    #                 pred_liks_ml.append(0)
    #                 pred_liks_ml_std.append(0)
    #             #VI
    #             try:
    #                 with open('results/movielens/VI_1000_{}_K{}_False.pkl'.format(lrs[lr],Ks[K]), 'rb') as f:
    #                     results_adam_tmc_new = pickle.load(f)
    #                 #moments
    #                 final_pred_lik_adam = results_adam_tmc_new['final_pred_lik_for_K'].mean(axis=0)
    #                 final_pred_lik_std_err_adam = results_adam_tmc_new['final_pred_lik_for_K'].std(axis=0) / np.sqrt(10)
    #
    #                 pred_liks_adam.append(final_pred_lik_adam[K])
    #                 pred_liks_adam_std.append(final_pred_lik_std_err_adam[K])
    #
    #             except:
    #                 pred_liks_adam.append(0)
    #                 pred_liks_adam_std.append(0)
    #
    #
    #
    #         offset = width * multiplier
    #         if K>0 and K_2>0:
    #             rects = ax[K,K_2].bar(x + offset, pred_liks_ml, width, yerr=pred_liks_ml_std, color=ml_colours[0])
    #             # ax_bar.bar_label(rects, padding=3)
    #             multiplier += 1
    #
    #             offset = width * multiplier
    #             rects = ax[K,K_2].bar(x + offset, pred_liks_adam, width, yerr=pred_liks_adam_std, color=adam_colours[0])
    #             # ax_bar.bar_label(rects, padding=3)
    #             multiplier += 1
    #         else:
    #             rects = ax[K,K_2].bar(x + offset, pred_liks_ml, width, yerr=pred_liks_ml_std,color=ml_colours[0], label='ML')
    #             # ax_bar.bar_label(rects, padding=3)
    #             multiplier += 1
    #
    #             offset = width * multiplier
    #             rects = ax[K,K_2].bar(x + offset, pred_liks_adam, width, yerr=pred_liks_adam_std, color=adam_colours[0], label='MP VI')
    #             # ax_bar.bar_label(rects, padding=3)
    #             multiplier += 1
    #
    #         ax[K,K_2].set_xticks(x + width, lrs)
    #         ax[2,1].set_xlabel('Evaluation K')
    #
    #         ax[1,0].set_ylabel(r'Training K')
    #
    #         ax[K,K_2].set_title(f'Training K: {Ks[K]}, Evaluation K: {Ks[K_2]}')
    #
    # blue_patch = mpatches.Patch(color=ml_colours[0], label='ML')
    # red_patch = mpatches.Patch(color=adam_colours[0], label='MP VI')
    # fig.legend(handles = [blue_patch, red_patch], loc='lower center', ncol=6, bbox_to_anchor=(0.55, -0.02))
    # # ax[3,K].legend(loc='lower center', bbox_to_anchor=(0.55, -0.15),
    # #                ncol=5)
    # fig.savefig('charts/chart_movielens_Kplot_synthetic.png')
    # fig.savefig('charts/chart_movielens_Kplot_synthetic.pdf')
