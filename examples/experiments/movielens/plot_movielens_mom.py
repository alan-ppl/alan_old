import json
import matplotlib.pyplot as plt
from tueplots import axes, bundles

# TODO: Clean this up, some of these functions can probably pretty easily be combined together


Ks = {"tmc_new": ['1', '3', '10', '30'], "global_k": ['1', '3','10','30', '100', '300', '1000', '3000', '10000']}#, '30000']}
Ns = ['5','10']
Ms = ['50', '150','300']

Ns = ['20']
Ms = ['450']

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

                val_tmc_new = [results["tmc_new"][k]['mean'] for k in Ks["tmc_new"]]
                std_tmc_new  = [results["tmc_new"][k]['std_err'] for k in Ks["tmc_new"]]

                val_global_K = [results["global_k"][k]['mean'] for k in Ks["global_k"]]
                std_global_K  = [results["global_k"][k]['std_err'] for k in Ks["global_k"]]


                # ax[i,j].errorbar(Ks,val_MP, yerr=std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                # ax[i,j].errorbar(Ks,val_tmc, yerr=std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(Ks["tmc_new"],val_tmc_new, yerr=std_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP')
                ax[i,j].errorbar(Ks["global_k"],val_global_K, yerr=std_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

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

                time_tmc_new = [results["tmc_new"][k]['time_mean'] for k in Ks["tmc_new"]]
                time_std_tmc_new  = [results["tmc_new"][k]['time_std_err'] for k in Ks["tmc_new"]]

                time_global_K = [results["global_k"][k]['time_mean'] for k in Ks["global_k"]]
                time_std_global_K = [results["global_k"][k]['time_std_err'] for k in Ks["global_k"]]


                # ax[i,j].errorbar(Ks,time_MP, yerr=time_std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                # ax[i,j].errorbar(Ks,time_tmc, yerr=time_std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(Ks["tmc_new"],time_tmc_new, yerr=time_std_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP')
                ax[i,j].errorbar(Ks["global_k"],time_global_K, yerr=time_std_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

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

                val_tmc_new = [results["tmc_new"][k]['mean'] for k in Ks["tmc_new"]]
                std_tmc_new  = [results["tmc_new"][k]['std_err'] for k in Ks["tmc_new"]]

                val_global_K = [results["global_k"][k]['mean'] for k in Ks["global_k"]]
                std_global_K  = [results["global_k"][k]['std_err'] for k in Ks["global_k"]]

                # time_MP = [results["MP"][k]['time_mean'] for k in Ks]
                # time_std_MP = [results["MP"][k]['time_std_err'] for k in Ks]

                # time_tmc = [results["tmc"][k]['time_mean'] for k in Ks]
                # time_std_tmc  = [results["tmc"][k]['time_std_err'] for k in Ks]

                time_tmc_new = [results["tmc_new"][k]['time_mean'] for k in Ks["tmc_new"]]
                time_std_tmc_new  = [results["tmc_new"][k]['time_std_err'] for k in Ks["tmc_new"]]

                time_global_K = [results["global_k"][k]['time_mean'] for k in Ks["global_k"]]
                time_std_global_K = [results["global_k"][k]['time_std_err'] for k in Ks["global_k"]]

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


                val_tmc_new = [results["tmc_new"][k][rv]["mean_var"] for k in Ks["tmc_new"]]
                val_global_K = [results["global_k"][k][rv]["mean_var"] for k in Ks["global_k"]]

                # ax[i,j].errorbar(Ks,val_MP, yerr=std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                # ax[i,j].errorbar(Ks,val_tmc, yerr=std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(Ks["tmc_new"],val_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP')
                ax[i,j].errorbar(Ks["global_k"],val_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

                count =+ 1
        # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
        ax[0,0].set_title('Number of users = 50')
        ax[0,1].set_title('Number of users = 150')
        ax[0,2].set_title('Number of users = 300')

        ylab = f"Mean Squared Error of \n{rv} Moment Estimator" if useData else f"Variance of {rv} Estimator"
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

                val_tmc_new = [results["tmc_new"][k][rv]["mean_var"] for k in Ks["tmc_new"]]
                val_global_K = [results["global_k"][k][rv]["mean_var"] for k in Ks["global_k"]]

                # time_MP = [results["MP"][k]['time_mean'] for k in Ks]
                # time_std_MP = [results["MP"][k]['time_std_err'] for k in Ks]

                # time_tmc = [results["tmc"][k]['time_mean'] for k in Ks]
                # time_std_tmc  = [results["tmc"][k]['time_std_err'] for k in Ks]

                time_tmc_new = [results["tmc_new"][k]['time_mean'] for k in Ks["tmc_new"]]
                time_std_tmc_new  = [results["tmc_new"][k]['time_std_err'] for k in Ks["tmc_new"]]

                time_global_K = [results["global_k"][k]['time_mean'] for k in Ks["global_k"]]
                time_std_global_K = [results["global_k"][k]['time_std_err'] for k in Ks["global_k"]]

                # ax[i,j].errorbar(time_MP,val_MP, yerr=std_MP, xerr=time_std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                # ax[i,j].errorbar(time_tmc,val_tmc, yerr=std_tmc, xerr=time_std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(time_tmc_new,val_tmc_new, xerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP')
                ax[i,j].errorbar(time_global_K,val_global_K, xerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

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

        ylab = f"Mean Squared Error of \n{rv} Moment Estimator" if useData else f"Variance of {rv} Estimator"
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

def plotAllForSinglePlateCombination(Ks, N, M, rv):
    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context(bundles.icml2022()):
        
        fig, ax = plt.subplots(2,4,figsize=(8.5, 4.5))
        for i in range(2):
            with open(f'results/movielens_elbo_N{N}_M{M}.json') as f:
                elbos = json.load(f)
            with open(f'results/movielens_p_ll_N{N}_M{M}.json') as f:
                p_lls = json.load(f)
            with open(f'results/movielens_variance_N{N}_M{M}.json') as f:
                vars = json.load(f)
            with open(f'results/movielens_MSE_N{N}_M{M}.json') as f:
                mses = json.load(f)

            elbo_MP = [elbos["tmc_new"][k]['mean'] for k in Ks["tmc_new"]]
            elbo_std_err_MP  = [elbos["tmc_new"][k]['std_err'] for k in Ks["tmc_new"]]
            elbo_time_MP = [elbos["tmc_new"][k]['time_mean'] for k in Ks["tmc_new"]]

            elbo_global_K = [elbos["global_k"][k]['mean'] for k in Ks["global_k"]]
            elbo_std_err_global_K  = [elbos["global_k"][k]['std_err'] for k in Ks["global_k"]]
            elbo_time_global_K = [elbos["global_k"][k]['time_mean'] for k in Ks["global_k"]]


            p_ll_MP = [p_lls["tmc_new"][k]['mean'] for k in Ks["tmc_new"]]
            p_ll_std_err_MP  = [p_lls["tmc_new"][k]['std_err'] for k in Ks["tmc_new"]]
            p_ll_time_MP = [p_lls["tmc_new"][k]['time_mean'] for k in Ks["tmc_new"]]


            p_ll_global_K = [p_lls["global_k"][k]['mean'] for k in Ks["global_k"]]
            p_ll_std_err_global_K  = [p_lls["global_k"][k]['std_err'] for k in Ks["global_k"]]
            p_ll_time_global_K = [mses["global_k"][k]['time_mean'] for k in Ks["global_k"]]

            vars_MP = [vars["tmc_new"][k][rv]['mean_var'] for k in Ks["tmc_new"]]
            vars_time_MP = [vars["tmc_new"][k]['time_mean'] for k in Ks["tmc_new"]]

            vars_global_K = [vars["global_k"][k][rv]['mean_var'] for k in Ks["global_k"]]
            vars_time_global_K = [vars["global_k"][k]['time_mean'] for k in Ks["global_k"]]

            mses_MP = [mses["tmc_new"][k][rv]['mean_var'] for k in Ks["tmc_new"]]
            mses_time_MP = [mses["tmc_new"][k]['time_mean'] for k in Ks["tmc_new"]]

            mses_global_K = [mses["global_k"][k][rv]['mean_var'] for k in Ks["global_k"]]
            mses_time_global_K = [mses["global_k"][k]['time_mean'] for k in Ks["global_k"]]

            if i == 0:
                ax[i,0].errorbar(Ks["tmc_new"],elbo_MP, yerr=elbo_std_err_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,0].errorbar(Ks["global_k"],elbo_global_K, yerr=elbo_std_err_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

                ax[i,1].errorbar(Ks["tmc_new"],p_ll_MP, yerr=p_ll_std_err_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,1].errorbar(Ks["global_k"],p_ll_global_K, yerr=p_ll_std_err_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

                ax[i,2].errorbar(Ks["tmc_new"],vars_MP, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,2].errorbar(Ks["global_k"],vars_global_K, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

                ax[i,3].errorbar(Ks["tmc_new"],mses_MP, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,3].errorbar(Ks["global_k"],mses_global_K, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

            else:
                ax[i,0].errorbar(elbo_time_MP,elbo_MP, yerr=elbo_std_err_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,0].errorbar(elbo_time_global_K,elbo_global_K, yerr=elbo_std_err_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

                ax[i,1].errorbar(p_ll_time_MP,p_ll_MP, yerr=p_ll_std_err_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,1].errorbar(p_ll_time_global_K,p_ll_global_K, yerr=p_ll_std_err_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

                ax[i,2].errorbar(vars_time_MP,vars_MP, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,2].errorbar(vars_time_global_K,vars_global_K, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

                ax[i,3].errorbar(mses_time_MP,mses_MP, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,3].errorbar(mses_time_global_K,mses_global_K, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

        for j in range(4):
            colTitles = ['a','b','c','d']
            ax[0,j].set_xlabel('K')
            ax[1,j].set_xlabel('Time (s)')

            ax[0,j].tick_params(axis='x', labelrotation = 90)

            ax[0,j].set_title(f'({colTitles[j]})', loc='left', weight="bold")

        for i in range(2):
            ax[i,0].set_ylabel("Evidence Lower Bound", fontsize=10)
            ax[i,1].set_ylabel("Predictive Log-Likelihood", fontsize=10)
            ax[i,2].set_ylabel(f"Variance of {rv} Estimator", fontsize=10)
            ax[i,3].set_ylabel(f"MSE of {rv} Estimator", fontsize=10)

        plt.legend()
        plt.savefig(f'plots/movielens_all_N{N}_M{M}_{rv}.png')
        plt.savefig(f'plots/movielens_all_N{N}_M{M}_{rv}.pdf')
        plt.close()

# plotAllForSinglePlateCombination(Ks, 10, 450, "z")
plotAllForSinglePlateCombination(Ks, 20, 450, "z")

# for useData in [True, False]:
#     if useData:
#         plotsFolder = "plots/trueData"
#         resultsFolder = "results/trueData"
#     else:
#         plotsFolder = "plots/sampledData"
#         resultsFolder = "results/sampledData"

#     multiPlotValsVsK(Ks, Ns, Ms, "elbo")
#     multiPlotValsVsK(Ks, Ns, Ms, "p_ll")

#     multiPlotTimeVsK(Ks, Ns, Ms, "elbo")
#     multiPlotTimeVsK(Ks, Ns, Ms, "p_ll")
#     multiPlotTimeVsK(Ks, Ns, Ms, "expectation")

#     multiPlotValsVsTime(Ks, Ns, Ms, "elbo")
#     multiPlotValsVsTime(Ks, Ns, Ms, "p_ll")

#     for rv in ["z", "psi_z", "mu_z"]:
#         multiPlotExpectationVarVsK(Ks, Ns, Ms, rv)
#         multiPlotEXpectationVarVsTime(Ks, Ns, Ms, rv)