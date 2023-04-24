import numpy as np
import matplotlib.pyplot as plt
import json
from tueplots import axes, bundles
from tueplots import cycler
from tueplots.constants import markers
from tueplots.constants.color import palettes

Ks_global = ['3','10','30','100','300', '1000', '3000', "10000", "30000"]
Ks_tmc = ['3','10','30']
# Ns = ['5','10']
# Ms = ['50','150','300']
Ns = ['2']
Ms = ['2']
lrs = ['0.01', '0.001', '0.0001', '1e-05', '1e-06']# '1e-07', '1e-08']
# with open('results.json') as f:
#     results = json.load(f)
#
# with open('results_local_IW.json') as f:
#     results_local_IW = json.load(f)

plt.rcParams.update({"figure.dpi": 300})
plt.rcParams.update(cycler.cycler(color=palettes.muted))
with plt.rc_context(bundles.icml2022()):

    count = 0
    fig, ax = plt.subplots(1,1, figsize=(5.5/2, 2.0))
    for i in range(len(Ns)):
        for j in range(len(Ms)):
            for lr in range(len(lrs)):
                N = Ns[i]
                M = Ms[j]
    #            colour =
                lr = lrs[lr]

                with open('results/bus_breakdown/ML_1_{}__N{}_M{}.json'.format(lr,N,M)) as f:
                    results_rws_tmc_new = json.load(f)

                # with open('results/movielens_discrete/rws_tmc_N{0}_M{1}.json'.format(N,M)) as f:
                #     results_rws_tmc = json.load(f)

                # with open('results/bus_breakdown/rws_tmc_new_N{0}_M{1}.json'.format(N,M)) as f:
                #     results_rws_tmc_new = json.load(f)

                # with open('results/movielens_discrete/rws_tmc_new_LIW_N{0}_M{1}.json'.format(N,M)) as f:
                #     results_IW = json.load(f)



                # elbos_rws_tmc = [results_rws_tmc[N][M][k]['pred_likelihood'] for k in Ks]
                # stds_rws_tmc = [results_rws_tmc[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]

                elbos_rws_tmc_new = [results_rws_tmc_new[N][M][k]['pred_likelihood'] for k in Ks_tmc] + [np.nan]*6
                stds_rws_tmc_new = [results_rws_tmc_new[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks_tmc] + [np.nan]*6
                time_rws_tmc_new = [results_rws_tmc_new[N][M][k]['avg_time'] for k in Ks_tmc] + [np.nan]*6
                stds_rws_tmc_new_time = [results_rws_tmc_new[N][M][k]['std_time']/np.sqrt(5) for k in Ks_tmc] + [np.nan]*6
                # elbos_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood'] for k in Ks_global]
                # stds_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks_global]

                # elbos_rws_IW = [results_IW[N][M][k]['pred_likelihood'] for k in Ks]
                # stds_rws_IW = [results_IW[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]

                #ax.errorbar(Ks_global,elbos_rws_global_k, yerr=stds_rws_global_k, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global RWS')
                # ax[i,j].errorbar(Ks,elbos_rws_tmc, yerr=stds_rws_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='orange', label='TMC RWS')
                ax.errorbar(time_rws_tmc_new,elbos_rws_tmc_new, yerr=stds_rws_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', label='lr: {}'.format(lr))
                # ax[i,j].errorbar(Ks,elbos_rws_IW, yerr=stds_rws_IW, linewidth=0.55, markersize = 0.75, fmt='-o', c='purple', label='RWS LIW')

                count =+ 1
    # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
    # ax[0,0].set_title('Number of users = 50')
    # ax[0,1].set_title('Number of users = 150')
    ax.set_title('1000 updates')
    ax.set_ylabel('Predictive Log Likelihood')
    # ax[1,0].set_ylabel('Films per user = 10 \n Predictive Log Likelihood')

    # ax[1,0].sharex(ax[0,0])
    ax.set_xlabel('Time (Seconds)')
    # ax.annotate(r'\bf{a}', xy=(0, 1), xycoords='axes fraction', fontsize=10,
    #             xytext=(-40, 5), textcoords='offset points',
    #             ha='right', va='bottom')
    # ax[1,1].sharex(ax[0,0])
    # ax[1,1].set_xlabel('K')
    # ax[1,2].sharex(ax[0,0])
    # ax[1,2].set_xlabel('K')
    # fig.tight_layout()
    # plt.legend()
    plt.legend()
    plt.savefig('charts/chart_bus_breakdown_predll_N{}_M{}.png'.format(N, M))
    plt.savefig('charts/chart_bus_breakdown_predll_N{}_M{}.pdf'.format(N, M))


plt.rcParams.update(cycler.cycler(color=palettes.muted))
with plt.rc_context(bundles.icml2022()):

    count = 0
    fig, ax = plt.subplots(1,1, figsize=(5.5/2, 2.0))
    for i in range(len(Ns)):
        for j in range(len(Ms)):
            for lr in range(len(lrs)):
                N = Ns[i]
                M = Ms[j]
    #            colour =
                lr = lrs[lr]

                with open('results/bus_breakdown/ML_1_{}__N{}_M{}.json'.format(lr,N,M)) as f:
                    results_rws_tmc_new = json.load(f)

                # with open('results/movielens_discrete/rws_tmc_N{0}_M{1}.json'.format(N,M)) as f:
                #     results_rws_tmc = json.load(f)

                # with open('results/bus_breakdown/rws_tmc_new_N{0}_M{1}.json'.format(N,M)) as f:
                #     results_rws_tmc_new = json.load(f)

                # with open('results/movielens_discrete/rws_tmc_new_LIW_N{0}_M{1}.json'.format(N,M)) as f:
                #     results_IW = json.load(f)



                # elbos_rws_tmc = [results_rws_tmc[N][M][k]['pred_likelihood'] for k in Ks]
                # stds_rws_tmc = [results_rws_tmc[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]

                elbos_rws_tmc_new = [results_rws_tmc_new[N][M][k]['final_obj'] for k in Ks_tmc] + [np.nan]*6
                stds_rws_tmc_new = [results_rws_tmc_new[N][M][k]['final_obj_std']/np.sqrt(5) for k in Ks_tmc] + [np.nan]*6
                time_rws_tmc_new = [results_rws_tmc_new[N][M][k]['avg_time'] for k in Ks_tmc] + [np.nan]*6
                stds_rws_tmc_new_time = [results_rws_tmc_new[N][M][k]['std_time']/np.sqrt(5) for k in Ks_tmc] + [np.nan]*6
                # elbos_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood'] for k in Ks_global]
                # stds_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks_global]

                # elbos_rws_IW = [results_IW[N][M][k]['pred_likelihood'] for k in Ks]
                # stds_rws_IW = [results_IW[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]

                #ax.errorbar(Ks_global,elbos_rws_global_k, yerr=stds_rws_global_k, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global RWS')
                # ax[i,j].errorbar(Ks,elbos_rws_tmc, yerr=stds_rws_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='orange', label='TMC RWS')
                ax.errorbar(time_rws_tmc_new,elbos_rws_tmc_new, yerr=stds_rws_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', label='lr: {}'.format(lr))
                # ax[i,j].errorbar(Ks,elbos_rws_IW, yerr=stds_rws_IW, linewidth=0.55, markersize = 0.75, fmt='-o', c='purple', label='RWS LIW')

                count =+ 1
    # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
    # ax[0,0].set_title('Number of users = 50')
    # ax[0,1].set_title('Number of users = 150')
    ax.set_title('1000 updates')
    ax.set_ylabel('Elbo')
    # ax[1,0].set_ylabel('Films per user = 10 \n Predictive Log Likelihood')

    # ax[1,0].sharex(ax[0,0])
    ax.set_xlabel('Time (Seconds)')
    # ax.annotate(r'\bf{a}', xy=(0, 1), xycoords='axes fraction', fontsize=10,
    #             xytext=(-40, 5), textcoords='offset points',
    #             ha='right', va='bottom')
    # ax[1,1].sharex(ax[0,0])
    # ax[1,1].set_xlabel('K')
    # ax[1,2].sharex(ax[0,0])
    # ax[1,2].set_xlabel('K')
    # fig.tight_layout()
    # plt.legend()
    plt.legend()
    plt.savefig('charts/chart_bus_breakdown_elbo_N{}_M{}.png'.format(N, M))
    plt.savefig('charts/chart_bus_breakdown_elbo_N{}_M{}.pdf'.format(N, M))

lrs = ['0.01']#, '0.001', '0.0001', '1e-05', '1e-06']# '1e-07', '1e-08']
# with open('results.json') as f:
#     results = json.load(f)
#
# with open('results_local_IW.json') as f:
#     results_local_IW = json.load(f)

plt.rcParams.update({"figure.dpi": 300})
plt.rcParams.update(cycler.cycler(color=palettes.muted))
with plt.rc_context(bundles.icml2022()):

    count = 0
    fig, ax = plt.subplots(1,1, figsize=(5.5/2, 2.0))
    for i in range(len(Ns)):
        for j in range(len(Ms)):
            for lr in range(len(lrs)):
                N = Ns[i]
                M = Ms[j]
    #            colour =
                lr = lrs[lr]

                with open('results/bus_breakdown/ML_1_{}__N{}_M{}.json'.format(lr,N,M)) as f:
                    results_rws_tmc_new = json.load(f)

                # with open('results/movielens_discrete/rws_tmc_N{0}_M{1}.json'.format(N,M)) as f:
                #     results_rws_tmc = json.load(f)

                # with open('results/bus_breakdown/rws_tmc_new_N{0}_M{1}.json'.format(N,M)) as f:
                #     results_rws_tmc_new = json.load(f)

                # with open('results/movielens_discrete/rws_tmc_new_LIW_N{0}_M{1}.json'.format(N,M)) as f:
                #     results_IW = json.load(f)



                # elbos_rws_tmc = [results_rws_tmc[N][M][k]['pred_likelihood'] for k in Ks]
                # stds_rws_tmc = [results_rws_tmc[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]

                elbos_rws_tmc_new = [results_rws_tmc_new[N][M][k]['pred_likelihood'] for k in Ks_tmc] + [np.nan]*6
                stds_rws_tmc_new = [results_rws_tmc_new[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks_tmc] + [np.nan]*6
                time_rws_tmc_new = [results_rws_tmc_new[N][M][k]['avg_time'] for k in Ks_tmc] + [np.nan]*6
                stds_rws_tmc_new_time = [results_rws_tmc_new[N][M][k]['std_time']/np.sqrt(5) for k in Ks_tmc] + [np.nan]*6
                # elbos_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood'] for k in Ks_global]
                # stds_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks_global]

                # elbos_rws_IW = [results_IW[N][M][k]['pred_likelihood'] for k in Ks]
                # stds_rws_IW = [results_IW[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]

                #ax.errorbar(Ks_global,elbos_rws_global_k, yerr=stds_rws_global_k, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global RWS')
                # ax[i,j].errorbar(Ks,elbos_rws_tmc, yerr=stds_rws_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='orange', label='TMC RWS')
                ax.errorbar(time_rws_tmc_new,elbos_rws_tmc_new, yerr=stds_rws_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', label='lr: {}'.format(lr))
                # ax[i,j].errorbar(Ks,elbos_rws_IW, yerr=stds_rws_IW, linewidth=0.55, markersize = 0.75, fmt='-o', c='purple', label='RWS LIW')

                count =+ 1
    # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
    # ax[0,0].set_title('Number of users = 50')
    # ax[0,1].set_title('Number of users = 150')
    ax.set_title('1000 updates')
    ax.set_ylabel('Predictive Log Likelihood')
    # ax[1,0].set_ylabel('Films per user = 10 \n Predictive Log Likelihood')

    # ax[1,0].sharex(ax[0,0])
    ax.set_xlabel('Time (Seconds)')
    # ax.annotate(r'\bf{a}', xy=(0, 1), xycoords='axes fraction', fontsize=10,
    #             xytext=(-40, 5), textcoords='offset points',
    #             ha='right', va='bottom')
    # ax[1,1].sharex(ax[0,0])
    # ax[1,1].set_xlabel('K')
    # ax[1,2].sharex(ax[0,0])
    # ax[1,2].set_xlabel('K')
    # fig.tight_layout()
    # plt.legend()
    plt.legend()
    plt.savefig('charts/chart_bus_breakdown_predll_only0.01_N{}_M{}.png'.format(N, M))
    plt.savefig('charts/chart_bus_breakdown_predll_only0.01_N{}_M{}.pdf'.format(N, M))


plt.rcParams.update(cycler.cycler(color=palettes.muted))
with plt.rc_context(bundles.icml2022()):

    count = 0
    fig, ax = plt.subplots(1,1, figsize=(5.5/2, 2.0))
    for i in range(len(Ns)):
        for j in range(len(Ms)):
            for lr in range(len(lrs)):
                N = Ns[i]
                M = Ms[j]
    #            colour =
                lr = lrs[lr]

                with open('results/bus_breakdown/ML_1_{}__N{}_M{}.json'.format(lr,N,M)) as f:
                    results_rws_tmc_new = json.load(f)

                # with open('results/movielens_discrete/rws_tmc_N{0}_M{1}.json'.format(N,M)) as f:
                #     results_rws_tmc = json.load(f)

                # with open('results/bus_breakdown/rws_tmc_new_N{0}_M{1}.json'.format(N,M)) as f:
                #     results_rws_tmc_new = json.load(f)

                # with open('results/movielens_discrete/rws_tmc_new_LIW_N{0}_M{1}.json'.format(N,M)) as f:
                #     results_IW = json.load(f)



                # elbos_rws_tmc = [results_rws_tmc[N][M][k]['pred_likelihood'] for k in Ks]
                # stds_rws_tmc = [results_rws_tmc[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]

                elbos_rws_tmc_new = [results_rws_tmc_new[N][M][k]['final_obj'] for k in Ks_tmc] + [np.nan]*6
                stds_rws_tmc_new = [results_rws_tmc_new[N][M][k]['final_obj_std']/np.sqrt(5) for k in Ks_tmc] + [np.nan]*6
                time_rws_tmc_new = [results_rws_tmc_new[N][M][k]['avg_time'] for k in Ks_tmc] + [np.nan]*6
                stds_rws_tmc_new_time = [results_rws_tmc_new[N][M][k]['std_time']/np.sqrt(5) for k in Ks_tmc] + [np.nan]*6
                # elbos_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood'] for k in Ks_global]
                # stds_rws_global_k = [results_rws_global_k[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks_global]

                # elbos_rws_IW = [results_IW[N][M][k]['pred_likelihood'] for k in Ks]
                # stds_rws_IW = [results_IW[N][M][k]['pred_likelihood_std']/np.sqrt(5) for k in Ks]

                #ax.errorbar(Ks_global,elbos_rws_global_k, yerr=stds_rws_global_k, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global RWS')
                # ax[i,j].errorbar(Ks,elbos_rws_tmc, yerr=stds_rws_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='orange', label='TMC RWS')
                ax.errorbar(time_rws_tmc_new,elbos_rws_tmc_new, yerr=stds_rws_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', label='lr: {}'.format(lr))
                # ax[i,j].errorbar(Ks,elbos_rws_IW, yerr=stds_rws_IW, linewidth=0.55, markersize = 0.75, fmt='-o', c='purple', label='RWS LIW')

                count =+ 1
    # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
    # ax[0,0].set_title('Number of users = 50')
    # ax[0,1].set_title('Number of users = 150')
    ax.set_title('1000 updates')
    ax.set_ylabel('Elbo')
    # ax[1,0].set_ylabel('Films per user = 10 \n Predictive Log Likelihood')

    # ax[1,0].sharex(ax[0,0])
    ax.set_xlabel('Time (Seconds)')
    # ax.annotate(r'\bf{a}', xy=(0, 1), xycoords='axes fraction', fontsize=10,
    #             xytext=(-40, 5), textcoords='offset points',
    #             ha='right', va='bottom')
    # ax[1,1].sharex(ax[0,0])
    # ax[1,1].set_xlabel('K')
    # ax[1,2].sharex(ax[0,0])
    # ax[1,2].set_xlabel('K')
    # fig.tight_layout()
    # plt.legend()
    plt.legend()
    plt.savefig('charts/chart_bus_breakdown_elbo_only0.01_N{}_M{}.png'.format(N, M))
    plt.savefig('charts/chart_bus_breakdown_elbo_only0.01_N{}_M{}.pdf'.format(N, M))
