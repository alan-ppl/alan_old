import json
import matplotlib.pyplot as plt
from tueplots import axes, bundles


def plotAllForSinglePlateCombination(Ks, experimentParams, rv):
    experimentName = experimentParams["experimentName"]
    plateSizeStr = "_".join([f"{plateName}{plateSize}" for plateName, plateSize in experimentParams["plateSizes"].items()])
    
    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context(bundles.icml2022()):
        
        with open(f'{experimentName}/results/{experimentName}_elbo_{plateSizeStr}.json') as f:
            elbos = json.load(f)
        with open(f'{experimentName}/results/{experimentName}_p_ll_{plateSizeStr}.json') as f:
            p_lls = json.load(f)
        with open(f'{experimentName}/results/{experimentName}_variance_{plateSizeStr}.json') as f:
            vars = json.load(f)
        with open(f'{experimentName}/results/{experimentName}_MSE_{plateSizeStr}.json') as f:
            mses = json.load(f)

        elbo_MP = [elbos["tmc_new"][k]['mean'] for k in Ks["tmc_new"]]
        elbo_std_err_MP  = [elbos["tmc_new"][k]['std_err'] for k in Ks["tmc_new"]]
        elbo_time_MP = [elbos["tmc_new"][k]['time_mean'] for k in Ks["tmc_new"]]

        elbo_global_K = [elbos["global_k"][k]['mean'] for k in Ks["global_k"]]
        elbo_std_err_global_K  = [elbos["global_k"][k]['std_err'] for k in Ks["global_k"]]
        elbo_time_global_K = [elbos["global_k"][k]['time_mean'] for k in Ks["global_k"]]


        p_ll_MP = [p_lls["tmc_new"][k]['mean'] for k in Ks["tmc_new"][1:]]
        p_ll_std_err_MP  = [p_lls["tmc_new"][k]['std_err'] for k in Ks["tmc_new"][1:]]
        p_ll_time_MP = [p_lls["tmc_new"][k]['time_mean'] for k in Ks["tmc_new"][1:]]


        p_ll_global_K = [p_lls["global_k"][k]['mean'] for k in Ks["global_k"][1:]]
        p_ll_std_err_global_K  = [p_lls["global_k"][k]['std_err'] for k in Ks["global_k"][1:]]
        p_ll_time_global_K = [p_lls["global_k"][k]['time_mean'] for k in Ks["global_k"][1:]]

        vars_MP = [vars["tmc_new"][k][rv]['mean_var'] for k in Ks["tmc_new"][1:]]
        vars_time_MP = [vars["tmc_new"][k]['time_mean'] for k in Ks["tmc_new"][1:]]

        vars_global_K = [vars["global_k"][k][rv]['mean_var'] for k in Ks["global_k"][1:]]
        vars_time_global_K = [vars["global_k"][k]['time_mean'] for k in Ks["global_k"][1:]]

        mses_MP = [mses["tmc_new"][k][rv]['mean_var'] for k in Ks["tmc_new"][1:]]
        mses_time_MP = [mses["tmc_new"][k]['time_mean'] for k in Ks["tmc_new"][1:]]

        mses_global_K = [mses["global_k"][k][rv]['mean_var'] for k in Ks["global_k"][1:]]
        mses_time_global_K = [mses["global_k"][k]['time_mean'] for k in Ks["global_k"][1:]]

        fig, ax = plt.subplots(2,4,figsize=(8.5, 4.5))
        for i in range(2):

            if i == 0:
                ax[i,0].errorbar(Ks["tmc_new"],elbo_MP, yerr=elbo_std_err_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,0].errorbar(Ks["global_k"],elbo_global_K, yerr=elbo_std_err_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

                ax[i,1].errorbar(Ks["tmc_new"][1:],p_ll_MP, yerr=p_ll_std_err_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,1].errorbar(Ks["global_k"][1:],p_ll_global_K, yerr=p_ll_std_err_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

                ax[i,2].errorbar(Ks["tmc_new"][1:],vars_MP, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,2].errorbar(Ks["global_k"][1:],vars_global_K, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

                ax[i,3].errorbar(Ks["tmc_new"][1:],mses_MP, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,3].errorbar(Ks["global_k"][1:],mses_global_K, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

            else:
                ax[i,0].errorbar(elbo_time_MP,elbo_MP, yerr=elbo_std_err_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,0].errorbar(elbo_time_global_K,elbo_global_K, yerr=elbo_std_err_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

                ax[i,1].errorbar(p_ll_time_MP,p_ll_MP, yerr=p_ll_std_err_MP, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,1].errorbar(p_ll_time_global_K,p_ll_global_K, yerr=p_ll_std_err_global_K, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

                ax[i,2].errorbar(vars_time_MP,vars_MP, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,2].errorbar(vars_time_global_K,vars_global_K, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

                ax[i,3].errorbar(mses_time_MP,mses_MP, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
                ax[i,3].errorbar(mses_time_global_K,mses_global_K, yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

        if rv == "alpha": rv = "IdMean"  # change to match the name of the variable in the paper

        for j in range(4):
            colTitles = ['a','b','c','d']
            ax[0,j].set_xlabel('K')
            ax[1,j].set_xlabel('Time (s)')

            ax[0,j].tick_params(axis='x', labelrotation = 90)

            # ax[0,j].set_title(f'({colTitles[j]})', loc='left', weight="bold")
        
        xColAnnotationCoords = {"movielens":     [-34, -34, -23, -22],
                                "bus_breakdown": [-37, -37, -26.7, -22.7],}

        ax[0,0].annotate(r'\bf{a}', xy=(0, 1), xycoords='axes fraction', fontsize=10,
            xytext=(xColAnnotationCoords[experimentName][0], 5), textcoords='offset points',
            ha='right', va='bottom')
        ax[0,1].annotate(r'\bf{b}', xy=(0, 1), xycoords='axes fraction', fontsize=10,
            xytext=(xColAnnotationCoords[experimentName][1], 5), textcoords='offset points',
            ha='right', va='bottom')
        ax[0,2].annotate(r'\bf{c}', xy=(0, 1), xycoords='axes fraction', fontsize=10,
            xytext=(xColAnnotationCoords[experimentName][2], 5), textcoords='offset points',
            ha='right', va='bottom')
        ax[0,3].annotate(r'\bf{d}', xy=(0, 1), xycoords='axes fraction', fontsize=10,
            xytext=(xColAnnotationCoords[experimentName][3], 5), textcoords='offset points',
            ha='right', va='bottom')

        for i in range(2):
            ax[i,0].set_ylabel("Evidence Lower Bound", fontsize=10)
            ax[i,1].set_ylabel("Predictive Log-Likelihood", fontsize=10)
            ax[i,2].set_ylabel(f"Variance of {rv} Estimator", fontsize=10)
            ax[i,3].set_ylabel(f"MSE of {rv} Estimator", fontsize=10)

        plt.legend()
        plt.savefig(f'plots/{experimentName}_all_{plateSizeStr}_{rv}.png')
        plt.savefig(f'plots/{experimentName}_all_{plateSizeStr}_{rv}.pdf')
        plt.close()

Ks = {"tmc_new": ['1', '3', '10', '30'], "global_k": ['1', '3','10','30', '100', '300', '1000', '3000', '10000']}#, '30000']}
plotAllForSinglePlateCombination(Ks, {"experimentName": "movielens", "plateSizes": {"N": 20, "M": 450,}}, "z")

Ks = {"tmc_new": ['1', '3', '10', '30'], "global_k": ['1', '3','10','30', '100', '300', '1000', '3000', '10000', '30000', '100000']}
plotAllForSinglePlateCombination(Ks, {"experimentName": "bus_breakdown", "plateSizes": {"M": 3, "J": 3, "I": 30}}, "alpha")

