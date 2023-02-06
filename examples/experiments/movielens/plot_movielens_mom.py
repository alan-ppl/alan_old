import json
import matplotlib.pyplot as plt
from tueplots import axes, bundles

# TODO: Clean this up, some of these functions can probably pretty easily be combined together

Ks = ['1', '3','10','30']
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
                    with open('results/movielens_elbo_N{0}_M{1}.json'.format(N,M)) as f:
                        results = json.load(f)
                elif mode == "p_ll":
                    with open('results/movielens_p_ll_N{0}_M{1}.json'.format(N,M)) as f:
                        results = json.load(f)


                val_MP = [results["MP"][k]['mean'] for k in Ks]
                std_MP  = [results["MP"][k]['std_err'] for k in Ks]

                val_tmc = [results["tmc"][k]['mean'] for k in Ks]
                std_tmc  = [results["tmc"][k]['std_err'] for k in Ks]

                val_tmc_new = [results["tmc_new"][k]['mean'] for k in Ks]
                std_tmc_new  = [results["tmc_new"][k]['std_err'] for k in Ks]

                val_global_K = [results["global_k"][k]['mean'] for k in Ks]
                std_global_K  = [results["global_k"][k]['std_err'] for k in Ks]


                ax[i,j].errorbar(Ks,val_MP, yerr=std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                ax[i,j].errorbar(Ks,val_tmc, yerr=std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(Ks,val_tmc_new, yerr=std_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='Permutation TMC')
                ax[i,j].errorbar(Ks,val_global_K, yerr=std_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

                # ax.set_title(f"Movielens N={N}, M={M}")
                # ax.set_ylabel("Evidence Lower Bound")
                # ax.set_xlabel("K")
                # plt.legend()
                # plt.savefig(f'plots/movielens_mom_N{N}_M{M}.png')
                # plt.savefig(f'plots/movielens_mom_N{N}_M{M}.pdf')


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
        plt.savefig('plots/movielens_mom_{0}.png'.format(mode))
        plt.savefig('plots/pdfs/movielens_mom_{0}.pdf'.format(mode))
        plt.close()

def multiPlotTimeVsK(Ks, Ns, Ms, mode="elbo"): # mode="elbo" or "p_ll" or "mean"
    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context(bundles.icml2022()):
        
        count = 0
        fig, ax = plt.subplots(2,3,figsize=(5.5, 3.5))
        for i in range(len(Ns)):
            for j in range(len(Ms)):
                
                N = Ns[i]
                M = Ms[j]
                
                if mode == "elbo":
                    with open('results/movielens_elbo_N{0}_M{1}.json'.format(N,M)) as f:
                        results = json.load(f)
                elif mode == "p_ll":
                    with open('results/movielens_p_ll_N{0}_M{1}.json'.format(N,M)) as f:
                        results = json.load(f)
                elif mode == "mean":
                    with open('results/movielens_mean_est_N{0}_M{1}.json'.format(N,M)) as f:
                        results = json.load(f)


                time_MP = [results["MP"][k]['time_mean'] for k in Ks]
                time_std_MP = [results["MP"][k]['time_std_err'] for k in Ks]

                time_tmc = [results["tmc"][k]['time_mean'] for k in Ks]
                time_std_tmc  = [results["tmc"][k]['time_std_err'] for k in Ks]

                time_tmc_new = [results["tmc_new"][k]['time_mean'] for k in Ks]
                time_std_tmc_new  = [results["tmc_new"][k]['time_std_err'] for k in Ks]

                time_global_K = [results["global_k"][k]['time_mean'] for k in Ks]
                time_std_global_K = [results["global_k"][k]['time_std_err'] for k in Ks]


                ax[i,j].errorbar(Ks,time_MP, yerr=time_std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                ax[i,j].errorbar(Ks,time_tmc, yerr=time_std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(Ks,time_tmc_new, yerr=time_std_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Permutation TMC')
                ax[i,j].errorbar(Ks,time_global_K, yerr=time_std_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='Global K')

                # ax.set_title(f"Movielens N={N}, M={M}")
                # ax.set_ylabel("Evidence Lower Bound")
                # ax.set_xlabel("K")
                # plt.legend()
                # plt.savefig(f'plots/movielens_mom_N{N}_M{M}.png')
                # plt.savefig(f'plots/movielens_mom_N{N}_M{M}.pdf')


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
        plt.savefig('plots/movielens_mom_{0}_time.png'.format(mode))
        plt.savefig('plots/pdfs/movielens_mom_{0}_time.pdf'.format(mode))
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
                    with open('results/movielens_elbo_N{0}_M{1}.json'.format(N,M)) as f:
                        results = json.load(f)
                elif mode == "p_ll":
                    with open('results/movielens_p_ll_N{0}_M{1}.json'.format(N,M)) as f:
                        results = json.load(f)


                val_MP = [results["MP"][k]['mean'] for k in Ks]
                std_MP  = [results["MP"][k]['std_err'] for k in Ks]

                val_tmc = [results["tmc"][k]['mean'] for k in Ks]
                std_tmc  = [results["tmc"][k]['std_err'] for k in Ks]

                val_tmc_new = [results["tmc_new"][k]['mean'] for k in Ks]
                std_tmc_new  = [results["tmc_new"][k]['std_err'] for k in Ks]

                val_global_K = [results["global_k"][k]['mean'] for k in Ks]
                std_global_K  = [results["global_k"][k]['std_err'] for k in Ks]


                time_MP = [results["MP"][k]['time_mean'] for k in Ks]
                time_std_MP = [results["MP"][k]['time_std_err'] for k in Ks]

                time_tmc = [results["tmc"][k]['time_mean'] for k in Ks]
                time_std_tmc  = [results["tmc"][k]['time_std_err'] for k in Ks]

                time_tmc_new = [results["tmc_new"][k]['time_mean'] for k in Ks]
                time_std_tmc_new  = [results["tmc_new"][k]['time_std_err'] for k in Ks]

                time_global_K = [results["global_k"][k]['time_mean'] for k in Ks]
                time_std_global_K = [results["global_k"][k]['time_std_err'] for k in Ks]

                ax[i,j].errorbar(time_MP,val_MP, yerr=std_MP, xerr=time_std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                ax[i,j].errorbar(time_tmc,val_tmc, yerr=std_tmc, xerr=time_std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(time_tmc_new,val_tmc_new, yerr=std_tmc_new, xerr=time_std_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='Permutation TMC')
                ax[i,j].errorbar(time_global_K,val_global_K, yerr=std_global_K, xerr=time_std_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

                # ax.set_title(f"Movielens N={N}, M={M}")
                # ax.set_ylabel("Evidence Lower Bound")
                # ax.set_xlabel("K")
                # plt.legend()
                # plt.savefig(f'plots/movielens_mom_N{N}_M{M}.png')
                # plt.savefig(f'plots/movielens_mom_N{N}_M{M}.pdf')


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
        plt.savefig('plots/movielens_mom_{0}_vs_time.png'.format(mode))
        plt.savefig('plots/pdfs/movielens_mom_{0}_vs_time.pdf'.format(mode))
        plt.close()


def multiPlotMeanNormVsK(Ks, Ns, Ms, rv=None, norm=None): # mode="elbo" or "p_ll" or "mean"; rv and norm used only if mode=="mean", 
    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context(bundles.icml2022()):
        
        count = 0
        fig, ax = plt.subplots(2,3,figsize=(5.5, 3.5))
        for i in range(len(Ns)):
            for j in range(len(Ms)):
                
                N = Ns[i]
                M = Ms[j]
                
                with open('results/movielens_mean_est_N{0}_M{1}.json'.format(N,M)) as f:
                    results = json.load(f)


                val_MP = [results["MP"][k][rv][f"{norm}_mean"] for k in Ks]
                std_MP  = [results["MP"][k][rv][f"{norm}_std_err"] for k in Ks]

                val_tmc = [results["tmc"][k][rv][f"{norm}_mean"]for k in Ks]
                std_tmc  = [results["tmc"][k][rv][f"{norm}_std_err"] for k in Ks]

                val_tmc_new = [results["tmc_new"][k][rv][f"{norm}_mean"] for k in Ks]
                std_tmc_new  = [results["tmc_new"][k][rv][f"{norm}_std_err"] for k in Ks]

                val_global_K = [results["global_k"][k][rv][f"{norm}_mean"] for k in Ks]
                std_global_K  = [results["global_k"][k][rv][f"{norm}_std_err"] for k in Ks]


                ax[i,j].errorbar(Ks,val_MP, yerr=std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                ax[i,j].errorbar(Ks,val_tmc, yerr=std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(Ks,val_tmc_new, yerr=std_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='Permutation TMC')
                ax[i,j].errorbar(Ks,val_global_K, yerr=std_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

                # ax.set_title(f"Movielens N={N}, M={M}")
                # ax.set_ylabel("Evidence Lower Bound")
                # ax.set_xlabel("K")
                # plt.legend()
                # plt.savefig(f'plots/movielens_mom_N{N}_M{M}.png')
                # plt.savefig(f'plots/movielens_mom_N{N}_M{M}.pdf')


                count =+ 1
        # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
        ax[0,0].set_title('Number of users = 50')
        ax[0,1].set_title('Number of users = 150')
        ax[0,2].set_title('Number of users = 300')

        ylab = f"Average {norm.capitalize()} Norm to Sample \n Mean of {rv} Expectation (s)"
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
        plt.savefig(f"plots/movielens_mean_{norm.capitalize()}_{rv}.png")
        plt.savefig(f"plots/pdfs/movielens_mean_{norm.capitalize()}_{rv}.pdf")
        plt.close()

def multiPlotMeanNormVsTime(Ks, Ns, Ms, rv=None, norm=None): # mode="elbo" or "p_ll" or "mean"; rv and norm used only if mode=="mean", 
    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context(bundles.icml2022()):
        
        count = 0
        fig, ax = plt.subplots(2,3,figsize=(5.5, 3.5))
        for i in range(len(Ns)):
            for j in range(len(Ms)):
                
                N = Ns[i]
                M = Ms[j]
                
                with open('results/movielens_mean_est_N{0}_M{1}.json'.format(N,M)) as f:
                    results = json.load(f)


                val_MP = [results["MP"][k][rv][f"{norm}_mean"] for k in Ks]
                std_MP  = [results["MP"][k][rv][f"{norm}_std_err"] for k in Ks]

                val_tmc = [results["tmc"][k][rv][f"{norm}_mean"]for k in Ks]
                std_tmc  = [results["tmc"][k][rv][f"{norm}_std_err"] for k in Ks]

                val_tmc_new = [results["tmc_new"][k][rv][f"{norm}_mean"] for k in Ks]
                std_tmc_new  = [results["tmc_new"][k][rv][f"{norm}_std_err"] for k in Ks]

                val_global_K = [results["global_k"][k][rv][f"{norm}_mean"] for k in Ks]
                std_global_K  = [results["global_k"][k][rv][f"{norm}_std_err"] for k in Ks]

                time_MP = [results["MP"][k]['time_mean'] for k in Ks]
                time_std_MP = [results["MP"][k]['time_std_err'] for k in Ks]

                time_tmc = [results["tmc"][k]['time_mean'] for k in Ks]
                time_std_tmc  = [results["tmc"][k]['time_std_err'] for k in Ks]

                time_tmc_new = [results["tmc_new"][k]['time_mean'] for k in Ks]
                time_std_tmc_new  = [results["tmc_new"][k]['time_std_err'] for k in Ks]

                time_global_K = [results["global_k"][k]['time_mean'] for k in Ks]
                time_std_global_K = [results["global_k"][k]['time_std_err'] for k in Ks]



                ax[i,j].errorbar(time_MP,val_MP, yerr=std_MP, xerr=time_std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                ax[i,j].errorbar(time_tmc,val_tmc, yerr=std_tmc, xerr=time_std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(time_tmc_new,val_tmc_new, yerr=std_tmc_new, xerr=time_std_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='Permutation TMC')
                ax[i,j].errorbar(time_global_K,val_global_K, yerr=std_global_K, xerr=time_std_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

                # ax.set_title(f"Movielens N={N}, M={M}")
                # ax.set_ylabel("Evidence Lower Bound")
                # ax.set_xlabel("K")
                # plt.legend()
                # plt.savefig(f'plots/movielens_mom_N{N}_M{M}.png')
                # plt.savefig(f'plots/movielens_mom_N{N}_M{M}.pdf')


                count =+ 1
        # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
        ax[0,0].set_title('Number of users = 50')
        ax[0,1].set_title('Number of users = 150')
        ax[0,2].set_title('Number of users = 300')

        ylab = f"Average {norm.capitalize()} Norm to Sample \n Mean of {rv} Expectation (s)"
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
        plt.savefig(f"plots/movielens_mean_{norm.capitalize()}_{rv}_vs_time.png")
        plt.savefig(f"plots/pdfs/movielens_mean_{norm.capitalize()}_{rv}_vs_time.pdf")
        plt.close()

def multiPlotNormTime(Ks, Ns, Ms, rv=None, norm=None): # mode="elbo" or "p_ll" or "mean"; rv and norm used only if mode=="mean", 
    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context(bundles.icml2022()):
        
        count = 0
        fig, ax = plt.subplots(2,3,figsize=(5.5, 3.5))
        for i in range(len(Ns)):
            for j in range(len(Ms)):
                
                N = Ns[i]
                M = Ms[j]
                
                with open('results/movielens_mean_est_N{0}_M{1}.json'.format(N,M)) as f:
                    results = json.load(f)


                val_MP = [results["MP"][k][rv][f"{norm}_mean"] for k in Ks]
                std_MP  = [results["MP"][k][rv][f"{norm}_std_err"] for k in Ks]

                val_tmc = [results["tmc"][k][rv][f"{norm}_mean"]for k in Ks]
                std_tmc  = [results["tmc"][k][rv][f"{norm}_std_err"] for k in Ks]

                val_tmc_new = [results["tmc_new"][k][rv][f"{norm}_mean"] for k in Ks]
                std_tmc_new  = [results["tmc_new"][k][rv][f"{norm}_std_err"] for k in Ks]

                val_global_K = [results["global_k"][k][rv][f"{norm}_mean"] for k in Ks]
                std_global_K  = [results["global_k"][k][rv][f"{norm}_std_err"] for k in Ks]

                time_MP = [results["MP"][k]['time_mean'] for k in Ks]
                time_std_MP = [results["MP"][k]['time_std_err'] for k in Ks]

                time_tmc = [results["tmc"][k]['time_mean'] for k in Ks]
                time_std_tmc  = [results["tmc"][k]['time_std_err'] for k in Ks]

                time_tmc_new = [results["tmc_new"][k]['time_mean'] for k in Ks]
                time_std_tmc_new  = [results["tmc_new"][k]['time_std_err'] for k in Ks]

                time_global_K = [results["global_k"][k]['time_mean'] for k in Ks]
                time_std_global_K = [results["global_k"][k]['time_std_err'] for k in Ks]



                ax[i,j].errorbar(Ks, time_MP, yerr=time_std_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                ax[i,j].errorbar(Ks, time_tmc, yerr=time_std_tmc, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(Ks, time_tmc_new, yerr=time_std_tmc_new, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='Permutation TMC')
                ax[i,j].errorbar(Ks, time_global_K, yerr=time_std_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

                # ax.set_title(f"Movielens N={N}, M={M}")
                # ax.set_ylabel("Evidence Lower Bound")
                # ax.set_xlabel("K")
                # plt.legend()
                # plt.savefig(f'plots/movielens_mom_N{N}_M{M}.png')
                # plt.savefig(f'plots/movielens_mom_N{N}_M{M}.pdf')


                count =+ 1
        # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
        ax[0,0].set_title('Number of users = 50')
        ax[0,1].set_title('Number of users = 150')
        ax[0,2].set_title('Number of users = 300')

        ylab = f"Time to compute expectations (s)"
        ax[0,0].set_ylabel(f'Films per user = 5 \n {ylab}')
        ax[1,0].set_ylabel(f'Films per user = 10 \n {ylab}')

        # ax[1,0].sharex(ax[0,0])
        ax[1,0].set_xlabel('K')
        # ax[1,1].sharex(ax[0,0])
        ax[1,1].set_xlabel('K')
        # ax[1,2].sharex(ax[0,0])
        ax[1,2].set_xlabel('K')
        # fig.tight_layout()
        plt.legend()
        plt.savefig(f"plots/movielens_{norm.capitalize()}_{rv}_time.png")
        plt.savefig(f"plots/pdfs/movielens_{norm.capitalize()}_{rv}_time.pdf")
        plt.close()

def multiPlotStdErrNormVsK(Ks, Ns, Ms, rv=None, norm=None): # mode="elbo" or "p_ll" or "mean"; rv and norm used only if mode=="mean", 
    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context(bundles.icml2022()):
        
        count = 0
        fig, ax = plt.subplots(2,3,figsize=(5.5, 3.5))
        for i in range(len(Ns)):
            for j in range(len(Ms)):
                
                N = Ns[i]
                M = Ms[j]
                
                with open('results/movielens_mean_est_N{0}_M{1}.json'.format(N,M)) as f:
                    results = json.load(f)


                val_MP = [results["MP"][k][rv][f"{norm}_mean"] for k in Ks]
                std_MP  = [results["MP"][k][rv][f"{norm}_std_err"] for k in Ks]

                val_tmc = [results["tmc"][k][rv][f"{norm}_mean"]for k in Ks]
                std_tmc  = [results["tmc"][k][rv][f"{norm}_std_err"] for k in Ks]

                val_tmc_new = [results["tmc_new"][k][rv][f"{norm}_mean"] for k in Ks]
                std_tmc_new  = [results["tmc_new"][k][rv][f"{norm}_std_err"] for k in Ks]

                val_global_K = [results["global_k"][k][rv][f"{norm}_mean"] for k in Ks]
                std_global_K  = [results["global_k"][k][rv][f"{norm}_std_err"] for k in Ks]


                ax[i,j].errorbar(Ks,std_MP, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='black', label='MP (old)')
                ax[i,j].errorbar(Ks,std_tmc, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='green', label='TMC')
                ax[i,j].errorbar(Ks,std_tmc_new, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='Permutation TMC')
                ax[i,j].errorbar(Ks,std_global_K, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global K')

                # ax.set_title(f"Movielens N={N}, M={M}")
                # ax.set_ylabel("Evidence Lower Bound")
                # ax.set_xlabel("K")
                # plt.legend()
                # plt.savefig(f'plots/movielens_mom_N{N}_M{M}.png')
                # plt.savefig(f'plots/pdfs/movielens_mom_N{N}_M{M}.pdf')


                count =+ 1
        # plt.title('Groups: 0, Observations per group: 1, with one standard deviation')
        ax[0,0].set_title('Number of users = 50')
        ax[0,1].set_title('Number of users = 150')
        ax[0,2].set_title('Number of users = 300')

        ylab = f"Std. Err of {norm.capitalize()} Norm to Sample \n Mean of {rv} Expectation (s)"
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
        plt.savefig(f"plots/movielens_std_err_{norm.capitalize()}_{rv}.png")
        plt.savefig(f"plots/pdfs/movielens_std_err_{norm.capitalize()}_{rv}.pdf")
        plt.close()


multiPlotValsVsK(Ks, Ns, Ms, "elbo")
multiPlotValsVsK(Ks, Ns, Ms, "p_ll")
multiPlotTimeVsK(Ks, Ns, Ms, "elbo")
multiPlotTimeVsK(Ks, Ns, Ms, "p_ll")
multiPlotTimeVsK(Ks, Ns, Ms, "mean")

multiPlotValsVsTime(Ks, Ns, Ms, "elbo")
multiPlotValsVsTime(Ks, Ns, Ms, "p_ll")

for norm in ["l1", "l2", "linf"]:
    for rv in ["z", "psi_z", "mu_z"]:
        multiPlotMeanNormVsK(Ks, Ns, Ms, rv, norm)
        multiPlotStdErrNormVsK(Ks, Ns, Ms, rv, norm)
        multiPlotMeanNormVsTime(Ks, Ns, Ms, rv, norm)
        multiPlotNormTime(Ks, Ns, Ms, rv, norm)