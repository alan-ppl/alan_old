import json
import matplotlib.pyplot as plt
from tueplots import axes, bundles

# TODO: Clean this up, some of these functions can probably pretty easily be combined together


Ks = ['1', '3','10','30']#, '50']
Ns = ['5','10']
Ms = ['50', '150','300']

def multiPlotValsVsK(Ks, Ns, Ms, mode="elbo"): # mode="elbo" or "p_ll"
    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context(bundles.icml2022()):
        
        count = 0
        fig, ax = plt.subplots(2,3,figsize=(5.5, 3.5))
        for i in range(len(Ns)):
            for j in range(len(Ms)):
                
                N = Ns[i]
                M = Ms[j]
                
                if mode == "elbo":
                    with open(f'{resultsFolder}/movielens_elbo_N{N}_M{M}.json') as f:
                        results = json.load(f)
                elif mode == "p_ll":
                    with open(f'{resultsFolder}/movielens_p_ll_N{N}_M{M}.json') as f:
                        results = json.load(f)


                # val_MP = [results["MP"][k]['mean'] for k in Ks]
                # std_MP  = [results["MP"][k]['std_err'] for k in Ks]

                # val_tmc = [results["tmc"][k]['mean'] for k in Ks]
                # std_tmc  = [results["tmc"][k]['std_err'] for k in Ks]

                val_tmc_new = [results["tmc_new"][k]['mean'] for k in Ks]
                std_tmc_new  = [results["tmc_new"][k]['std_err'] for k in Ks]

                val_global_K = [results["global_k"][k]['mean'] for k in Ks]
                std_global_K  = [results["global_k"][k]['std_err'] for k in Ks]


                # ax[i,j].errorbar(Ks,val_MP, yerr=std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                # ax[i,j].errorbar(Ks,val_tmc, yerr=std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(Ks,val_tmc_new, yerr=std_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP')
                ax[i,j].errorbar(Ks,val_global_K, yerr=std_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

                # ax.set_title(f"Movielens N={N}, M={M}")
                # ax.set_ylabel("Evidence Lower Bound")
                # ax.set_xlabel("K")
                # plt.legend()
                # plt.savefig(f'{plotsFolder}/movielens_mom_N{N}_M{M}.png')
                # plt.savefig(f'{plotsFolder}/movielens_mom_N{N}_M{M}.pdf')


                count =+ 1
        # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
        ax[0,0].set_title('Number of users = 50')
        ax[0,1].set_title('Number of users = 150')
        ax[0,2].set_title('Number of users = 300')

        ylab = "Evidence Lower Bound" if mode == "elbo" else "Predictive Log-Likelihood"
        ax[0,0].set_ylabel(f'Films per user = 5 \n {ylab}')
        ax[1,0].set_ylabel(f'Films per user = 10 \n {ylab}')

        ax[1,0].sharex(ax[0,0])
        ax[1,0].set_xlabel('K')
        ax[1,1].sharex(ax[0,0])
        ax[1,1].set_xlabel('K')
        ax[1,2].sharex(ax[0,0])
        ax[1,2].set_xlabel('K')
        # fig.tight_layout()
        plt.legend()
        plt.savefig(f'{plotsFolder}/movielens_mom_{mode}.png')
        plt.savefig(f'{plotsFolder}/pdfs/movielens_mom_{mode}.pdf')
        plt.close()

def multiPlotTimeVsK(Ks, Ns, Ms, mode="elbo"): # mode="elbo" or "p_ll" or "expectation"
    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context(bundles.icml2022()):
        
        count = 0
        fig, ax = plt.subplots(2,3,figsize=(5.5, 3.5))
        for i in range(len(Ns)):
            for j in range(len(Ms)):
                
                N = Ns[i]
                M = Ms[j]
                
                with open(f'{resultsFolder}/movielens_{mode}_N{N}_M{M}.json') as f:
                    results = json.load(f)


                # time_MP = [results["MP"][k]['time_mean'] for k in Ks]
                # time_std_MP = [results["MP"][k]['time_std_err'] for k in Ks]

                # time_tmc = [results["tmc"][k]['time_mean'] for k in Ks]
                # time_std_tmc  = [results["tmc"][k]['time_std_err'] for k in Ks]

                time_tmc_new = [results["tmc_new"][k]['time_mean'] for k in Ks]
                time_std_tmc_new  = [results["tmc_new"][k]['time_std_err'] for k in Ks]

                time_global_K = [results["global_k"][k]['time_mean'] for k in Ks]
                time_std_global_K = [results["global_k"][k]['time_std_err'] for k in Ks]


                # ax[i,j].errorbar(Ks,time_MP, yerr=time_std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                # ax[i,j].errorbar(Ks,time_tmc, yerr=time_std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(Ks,time_tmc_new, yerr=time_std_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP')
                ax[i,j].errorbar(Ks,time_global_K, yerr=time_std_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

                # ax.set_title(f"Movielens N={N}, M={M}")
                # ax.set_ylabel("Evidence Lower Bound")
                # ax.set_xlabel("K")
                # plt.legend()
                # plt.savefig(f'{plotsFolder}/movielens_mom_N{N}_M{M}.png')
                # plt.savefig(f'{plotsFolder}/movielens_mom_N{N}_M{M}.pdf')


                count =+ 1
        # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
        ax[0,0].set_title('Number of users = 50')
        ax[0,1].set_title('Number of users = 150')
        ax[0,2].set_title('Number of users = 300')

        ylab = "Evidence Lower Bound" if mode == "elbo" else "Predictive Log-Likelihood" if mode == "p_ll" else "Expectations"
        ax[0,0].set_ylabel(f'Films per user = 5 \n Time to Compute \n {ylab} (s)')
        ax[1,0].set_ylabel(f'Films per user = 10 \n Time to Compute \n {ylab} (s)')

        ax[1,0].sharex(ax[0,0])
        ax[1,0].set_xlabel('K')
        ax[1,1].sharex(ax[0,0])
        ax[1,1].set_xlabel('K')
        ax[1,2].sharex(ax[0,0])
        ax[1,2].set_xlabel('K')
        # fig.tight_layout()
        plt.legend()
        plt.savefig(f'{plotsFolder}/movielens_mom_{mode}_time.png')
        plt.savefig(f'{plotsFolder}/pdfs/movielens_mom_{mode}_time.pdf')
        plt.close()

def multiPlotValsVsTime(Ks, Ns, Ms, mode="elbo"): # mode="elbo" or "p_ll"
    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context(bundles.icml2022()):
        
        count = 0
        fig, ax = plt.subplots(2,3,figsize=(5.5, 3.5))
        for i in range(len(Ns)):
            for j in range(len(Ms)):
                
                N = Ns[i]
                M = Ms[j]
                
                if mode == "elbo":
                    with open(f'{resultsFolder}/movielens_elbo_N{N}_M{M}.json') as f:
                        results = json.load(f)
                elif mode == "p_ll":
                    with open(f'{resultsFolder}/movielens_p_ll_N{N}_M{M}.json') as f:
                        results = json.load(f)


                # val_MP = [results["MP"][k]['mean'] for k in Ks]
                # std_MP  = [results["MP"][k]['std_err'] for k in Ks]

                # val_tmc = [results["tmc"][k]['mean'] for k in Ks]
                # std_tmc  = [results["tmc"][k]['std_err'] for k in Ks]

                val_tmc_new = [results["tmc_new"][k]['mean'] for k in Ks]
                std_tmc_new  = [results["tmc_new"][k]['std_err'] for k in Ks]

                val_global_K = [results["global_k"][k]['mean'] for k in Ks]
                std_global_K  = [results["global_k"][k]['std_err'] for k in Ks]

                # time_MP = [results["MP"][k]['time_mean'] for k in Ks]
                # time_std_MP = [results["MP"][k]['time_std_err'] for k in Ks]

                # time_tmc = [results["tmc"][k]['time_mean'] for k in Ks]
                # time_std_tmc  = [results["tmc"][k]['time_std_err'] for k in Ks]

                time_tmc_new = [results["tmc_new"][k]['time_mean'] for k in Ks]
                time_std_tmc_new  = [results["tmc_new"][k]['time_std_err'] for k in Ks]

                time_global_K = [results["global_k"][k]['time_mean'] for k in Ks]
                time_std_global_K = [results["global_k"][k]['time_std_err'] for k in Ks]

                # ax[i,j].errorbar(time_MP,val_MP, yerr=std_MP, xerr=time_std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                # ax[i,j].errorbar(time_tmc,val_tmc, yerr=std_tmc, xerr=time_std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(time_tmc_new,val_tmc_new, yerr=std_tmc_new, xerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP')
                ax[i,j].errorbar(time_global_K,val_global_K, yerr=std_global_K, xerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

                ax[i,j].set_xlim(xmin=0)
                # ax.set_title(f"Movielens N={N}, M={M}")
                # ax.set_ylabel("Evidence Lower Bound")
                # ax.set_xlabel("K")
                # plt.legend()
                # plt.savefig(f'{plotsFolder}/movielens_mom_N{N}_M{M}.png')
                # plt.savefig(f'{plotsFolder}/movielens_mom_N{N}_M{M}.pdf')


                count =+ 1
        # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')

        ax[0,0].set_title('Number of users = 50')
        ax[0,1].set_title('Number of users = 150')
        ax[0,2].set_title('Number of users = 300')

        ylab = "Evidence Lower Bound" if mode == "elbo" else "Predictive Log-Likelihood"
        ax[0,0].set_ylabel(f'Films per user = 5 \n {ylab}')
        ax[1,0].set_ylabel(f'Films per user = 10 \n {ylab}')

        # ax[1,0].sharex(ax[0,0])
        ax[1,0].set_xlabel('Time (s)')
        # ax[1,1].sharex(ax[0,0])
        ax[1,1].set_xlabel('Time (s)')
        # ax[1,2].sharex(ax[0,0])
        ax[1,2].set_xlabel('Time (s)')
        
        # fig.tight_layout()
        plt.legend()
        plt.savefig(f'{plotsFolder}/movielens_mom_{mode}_vs_time.png')
        plt.savefig(f'{plotsFolder}/pdfs/movielens_mom_{mode}_vs_time.pdf')
        plt.close()


def multiPlotExpectationVarVsK(Ks, Ns, Ms, rv):
    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context(bundles.icml2022()):
        
        count = 0
        fig, ax = plt.subplots(2,3,figsize=(5.5, 3.5))
        for i in range(len(Ns)):
            for j in range(len(Ms)):
                
                N = Ns[i]
                M = Ms[j]
                
                with open(f'{resultsFolder}/movielens_expectation_N{N}_M{M}.json') as f:
                    results = json.load(f)


                val_tmc_new = [results["tmc_new"][k][rv]["mean_var"] for k in Ks]
                val_global_K = [results["global_k"][k][rv]["mean_var"] for k in Ks]

                # ax[i,j].errorbar(Ks,val_MP, yerr=std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                # ax[i,j].errorbar(Ks,val_tmc, yerr=std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(Ks,val_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP')
                ax[i,j].errorbar(Ks,val_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

                count =+ 1
        # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
        ax[0,0].set_title('Number of users = 50')
        ax[0,1].set_title('Number of users = 150')
        ax[0,2].set_title('Number of users = 300')

        ylab = f"Mean Squared Error of \n{rv} Moment Estimator" if useData else f"Variance of {rv} Moment Estimator"
        ax[0,0].set_ylabel(f'Films per user = 5 \n {ylab}')
        ax[1,0].set_ylabel(f'Films per user = 10 \n {ylab}')

        ax[1,0].sharex(ax[0,0])
        ax[1,0].set_xlabel('K')
        ax[1,1].sharex(ax[0,0])
        ax[1,1].set_xlabel('K')
        ax[1,2].sharex(ax[0,0])
        ax[1,2].set_xlabel('K')
        # fig.tight_layout()
        plt.legend()
        plt.savefig(f"{plotsFolder}/movielens_expectation_{rv}.png")
        plt.savefig(f"{plotsFolder}/pdfs/movielens_expectation_{rv}.pdf")
        plt.close()

def multiPlotEXpectationVarVsTime(Ks, Ns, Ms, rv):
    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context(bundles.icml2022()):
        
        count = 0
        fig, ax = plt.subplots(2,3,figsize=(5.5, 3.5))
        for i in range(len(Ns)):
            for j in range(len(Ms)):
                
                N = Ns[i]
                M = Ms[j]
                
                with open(f'{resultsFolder}/movielens_expectation_N{N}_M{M}.json') as f:
                    results = json.load(f)


                # val_MP = [results["MP"][k][rv][f"{norm}_mean"] for k in Ks]
                # std_MP  = [results["MP"][k][rv][f"{norm}_std_err"] for k in Ks]

                # val_tmc = [results["tmc"][k][rv][f"{norm}_mean"]for k in Ks]
                # std_tmc  = [results["tmc"][k][rv][f"{norm}_std_err"] for k in Ks]

                val_tmc_new = [results["tmc_new"][k][rv]["mean_var"] for k in Ks]
                val_global_K = [results["global_k"][k][rv]["mean_var"] for k in Ks]

                # time_MP = [results["MP"][k]['time_mean'] for k in Ks]
                # time_std_MP = [results["MP"][k]['time_std_err'] for k in Ks]

                # time_tmc = [results["tmc"][k]['time_mean'] for k in Ks]
                # time_std_tmc  = [results["tmc"][k]['time_std_err'] for k in Ks]

                time_tmc_new = [results["tmc_new"][k]['time_mean'] for k in Ks]
                time_std_tmc_new  = [results["tmc_new"][k]['time_std_err'] for k in Ks]

                time_global_K = [results["global_k"][k]['time_mean'] for k in Ks]
                time_std_global_K = [results["global_k"][k]['time_std_err'] for k in Ks]

                # ax[i,j].errorbar(time_MP,val_MP, yerr=std_MP, xerr=time_std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                # ax[i,j].errorbar(time_tmc,val_tmc, yerr=std_tmc, xerr=time_std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(time_tmc_new,val_tmc_new, xerr=time_std_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP')
                ax[i,j].errorbar(time_global_K,val_global_K, xerr=time_std_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

                ax[i,j].set_xlim(xmin=0)
                # ax.set_title(f"Movielens N={N}, M={M}")
                # ax.set_ylabel("Evidence Lower Bound")
                # ax.set_xlabel("K")
                # plt.legend()
                # plt.savefig(f'{plotsFolder}/movielens_mom_N{N}_M{M}.png')
                # plt.savefig(f'{plotsFolder}/movielens_mom_N{N}_M{M}.pdf')


                count =+ 1
        # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
        ax[0,0].set_title('Number of users = 50')
        ax[0,1].set_title('Number of users = 150')
        ax[0,2].set_title('Number of users = 300')

        ylab = f"Mean Squared Error of \n{rv} Moment Estimator" if useData else f"Variance of {rv} Moment Estimator"
        ax[0,0].set_ylabel(f'Films per user = 5 \n {ylab}')
        ax[1,0].set_ylabel(f'Films per user = 10 \n {ylab}')

        # ax[1,0].sharex(ax[0,0])
        ax[1,0].set_xlabel('Time (s)')
        # ax[1,1].sharex(ax[0,0])
        ax[1,1].set_xlabel('Time (s)')
        # ax[1,2].sharex(ax[0,0])
        ax[1,2].set_xlabel('Time (s)')
        # fig.tight_layout()
        plt.legend()
        plt.savefig(f"{plotsFolder}/movielens_expectation_{rv}_vs_time.png")
        plt.savefig(f"{plotsFolder}/pdfs/movielens_expectation_vs_time.pdf")
        plt.close()

for useData in [True, False]:
    if useData:
        plotsFolder = "plots/trueData"
        resultsFolder = "results/trueData"
    else:
        plotsFolder = "plots/sampledData"
        resultsFolder = "results/sampledData"

    multiPlotValsVsK(Ks, Ns, Ms, "elbo")
    multiPlotValsVsK(Ks, Ns, Ms, "p_ll")

    multiPlotTimeVsK(Ks, Ns, Ms, "elbo")
    multiPlotTimeVsK(Ks, Ns, Ms, "p_ll")
    multiPlotTimeVsK(Ks, Ns, Ms, "expectation")

    multiPlotValsVsTime(Ks[:-1], Ns, Ms, "elbo")
    multiPlotValsVsTime(Ks[:-1], Ns, Ms, "p_ll")

    for rv in ["z", "psi_z", "mu_z"]:
        multiPlotExpectationVarVsK(Ks, Ns, Ms, rv)
        multiPlotEXpectationVarVsTime(Ks[:-1], Ns, Ms, rv)