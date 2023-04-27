import json
import matplotlib.pyplot as plt
from tueplots import axes, bundles
import combineJSONs

def combineVIJsons(vi_lrs, experimentName, plateSizeStr):
    for metric in ["elbo", "MSE", "p_ll", "variance"]:
        combineJSONs.combineMultiple(
            [f"{experimentName}/results/vi_{experimentName}_{metric}_{plateSizeStr}_lr{lr}.json" for lr in vi_lrs],
             f"{experimentName}/results/vi_{experimentName}_{metric}_{plateSizeStr}.json")
            
def combineIS_VIJsons( experimentName, plateSizeStr):
    for metric in ["elbo", "MSE", "p_ll", "variance"]:
        combineJSONs.combine(
            f"{experimentName}/results/{experimentName}_{metric}_{plateSizeStr}.json",
            f"{experimentName}/results/vi_{experimentName}_{metric}_{plateSizeStr}.json",
            f"{experimentName}/results/ALL_{experimentName}_{metric}_{plateSizeStr}.json")

def getSeriesValues(data, method, Ks, valueType, rv=None):
    if rv is None:
        return [data[method][k][valueType] for k in Ks[method]]
    else:
        return [data[method][k][rv][valueType] for k in Ks[method]]
    
def getMeanValues(data, method, Ks, rv=None):
    if rv is None:
        return getSeriesValues(data, method, Ks, "mean")
    else:
        return getSeriesValues(data, method, Ks, "mean_var", rv)
    
def getStdErrValues(data, method, Ks, rv=None):
    if rv is None:
        return getSeriesValues(data, method, Ks, "std_err")
    else:
        return []

def getTimeValues(data, method, Ks, rv=None):
    if rv is None:
        return getSeriesValues(data, method, Ks, "time_mean")
    else:
        return getSeriesValues(data, method, Ks, "time_mean")#, rv)
    

def plotAllForSinglePlateCombination(Ks, experimentParams, rv, vi_lrs=[0.1,0.01]):
    experimentName = experimentParams["experimentName"]
    plateSizeStr = "_".join([f"{plateName}{plateSize}" for plateName, plateSize in experimentParams["plateSizes"].items()])
    
    if len(vi_lrs) >= 1:
        combineVIJsons(vi_lrs, experimentName, plateSizeStr)
        combineIS_VIJsons(experimentName, plateSizeStr)

    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context(bundles.icml2022()):
        
        with open(f'{experimentName}/results/ALL_{experimentName}_elbo_{plateSizeStr}.json') as f:
            elbos = json.load(f)
        with open(f'{experimentName}/results/ALL_{experimentName}_p_ll_{plateSizeStr}.json') as f:
            p_lls = json.load(f)
        with open(f'{experimentName}/results/ALL_{experimentName}_variance_{plateSizeStr}.json') as f:
            vars_ = json.load(f)
        with open(f'{experimentName}/results/ALL_{experimentName}_MSE_{plateSizeStr}.json') as f:
            mses = json.load(f)

        results = {}
        metricToData = {"elbo": elbos, "MSE": mses, "p_ll": p_lls, "vars": vars_}
        for metric in metricToData: 
            data = metricToData[metric]
            results[metric] = {}
            for method in Ks:
                if metric in ("elbo", "p_ll"):
                    rv_temp = None
                else:
                    rv_temp = rv
                results[metric][method] = {"mean": getMeanValues(data, method, Ks, rv_temp),
                                           "std_err": getStdErrValues(data, method, Ks, rv_temp),
                                           "time": getTimeValues(data, method, Ks, rv_temp)}

        fig, ax = plt.subplots(2,4,figsize=(8.5, 4.5))
        
        ax[0,0].errorbar(Ks["tmc_new"],  results["elbo"]["tmc_new"]["mean"],  yerr=results["elbo"]["tmc_new"]["std_err"],  linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
        ax[0,0].errorbar(Ks["global_k"], results["elbo"]["global_k"]["mean"], yerr=results["elbo"]["global_k"]["std_err"], linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

        ax[0,1].errorbar(Ks["tmc_new"][1:],  results["p_ll"]["tmc_new"]["mean"][1:],  yerr=results["p_ll"]["tmc_new"]["std_err"][1:],  linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
        ax[0,1].errorbar(Ks["global_k"][1:], results["p_ll"]["global_k"]["mean"][1:], yerr=results["p_ll"]["global_k"]["std_err"][1:], linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

        ax[0,2].errorbar(Ks["tmc_new"][1:],  results["vars"]["tmc_new"]["mean"][1:],  yerr=0,  linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
        ax[0,2].errorbar(Ks["global_k"][1:], results["vars"]["global_k"]["mean"][1:], yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

        ax[0,3].errorbar(Ks["tmc_new"][1:],  results["MSE"]["tmc_new"]["mean"][1:],  yerr=0,  linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
        ax[0,3].errorbar(Ks["global_k"][1:], results["MSE"]["global_k"]["mean"][1:], yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')

        ax[1,0].errorbar(results["elbo"]["tmc_new"]["time"],  results["elbo"]["tmc_new"]["mean"],  yerr=results["elbo"]["tmc_new"]["std_err"],  linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
        ax[1,0].errorbar(results["elbo"]["global_k"]["time"], results["elbo"]["global_k"]["mean"], yerr=results["elbo"]["global_k"]["std_err"], linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')
        for lr in vi_lrs:
            if not(experimentName == "bus_breakdown" and lr in [0.1, 0.01]):
                ax[1,0].plot(results["elbo"][f"vi_{lr}"]["time"], results["elbo"][f"vi_{lr}"]["mean"], linewidth=0.55, markersize = 0.75, label="VI lr="+str(lr))
        # ax[1,0].set_yscale("log")

        ax[1,1].errorbar(results["p_ll"]["tmc_new"]["time"][1:],  results["p_ll"]["tmc_new"]["mean"][1:],  yerr=results["p_ll"]["tmc_new"]["std_err"][1:],  linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
        ax[1,1].errorbar(results["p_ll"]["global_k"]["time"][1:], results["p_ll"]["global_k"]["mean"][1:], yerr=results["p_ll"]["global_k"]["std_err"][1:], linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')
        for lr in vi_lrs:
            ax[1,1].plot(results["p_ll"][f"vi_{lr}"]["time"], results["p_ll"][f"vi_{lr}"]["mean"], linewidth=0.55, markersize = 0.75, label="VI lr="+str(lr))

        ax[1,2].errorbar(results["vars"]["tmc_new"]["time"][1:],  results["vars"]["tmc_new"]["mean"][1:],  yerr=0,  linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
        ax[1,2].errorbar(results["vars"]["global_k"]["time"][1:], results["vars"]["global_k"]["mean"][1:], yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')
        for lr in vi_lrs:
            ax[1,2].plot(results["vars"][f"vi_{lr}"]["time"], results["vars"][f"vi_{lr}"]["mean"], linewidth=0.55, markersize = 0.75, label="VI lr="+str(lr))

        ax[1,3].errorbar(results["MSE"]["tmc_new"]["time"][1:],  results["MSE"]["tmc_new"]["mean"][1:],  yerr=0,  linewidth=0.55, markersize = 0.75, fmt='-o', c='red', label='MP IS')
        ax[1,3].errorbar(results["MSE"]["global_k"]["time"][1:], results["MSE"]["global_k"]["mean"][1:], yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='blue', label='Global IS')
        for lr in vi_lrs:
            ax[1,3].plot(results["elbo"][f"vi_{lr}"]["time"], results["MSE"][f"vi_{lr}"]["mean"], linewidth=0.55, markersize = 0.75, label="VI lr="+str(lr))

        if rv == "alpha": rv = "IdMean"  # change to match the name of the variable in the paper

        for j in range(4):
            colTitles = ['a','b','c','d']
            ax[0,j].set_xlabel('K')
            ax[1,j].set_xlabel('Time (s)')

            ax[0,j].tick_params(axis='x', labelrotation = 90)

            # ax[0,j].set_title(f'({colTitles[j]})', loc='left', weight="bold")
        
        xColAnnotationCoords = {"movielens":     [-34, -34, -23, -22],
                                "bus_breakdown": [-37, -37, -23, -22.7],}

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

        vi_str = "" if len(vi_lrs) == 0 else "_vi"

        plt.legend()
        plt.savefig(f'plots/{experimentName}_all_{plateSizeStr}_{rv}{vi_str}.png')
        plt.savefig(f'plots/{experimentName}_all_{plateSizeStr}_{rv}{vi_str}.pdf')
        plt.close()


# MOVIELENS
vi_lrs= [0.0001, 0.001, 0.01, 0.1]

num_vi_iters = 100
vi_iter_step = 1
vi_iter_counts = [str(x) for x in range(0, num_vi_iters+1, vi_iter_step)]

Ks = {"tmc_new": ['1', '3', '10', '30'], "global_k": ['1', '3','10','30', '100', '300', '1000', '3000', '10000']}
# Ks = {"tmc_new": ['1', '3', '10'], "global_k": ['1', '3','10','30', '100', '300', '1000']}
for lr in vi_lrs:
    Ks[f"vi_{lr}"] = vi_iter_counts

plotAllForSinglePlateCombination(Ks, {"experimentName": "movielens", "plateSizes": {"N": 20, "M": 450,}}, "z", vi_lrs=vi_lrs)

plotAllForSinglePlateCombination(Ks, {"experimentName": "movielens", "plateSizes": {"N": 20, "M": 450,}}, "z", vi_lrs=[])

# BUS BREAKDOWN
vi_lrs=[0.001, 0.01, 0.1]

num_vi_iters = 100#200
vi_iter_step = 1
vi_iter_counts = [str(x) for x in range(0, num_vi_iters+1, vi_iter_step)]

Ks = {"tmc_new": ['1', '3', '10', '30'], "global_k": ['1', '3','10','30', '100', '300', '1000', '3000', '10000', '30000', '100000']}#, "vi": vi_iter_counts}
# Ks = {"tmc_new": ['1', '3', '10'], "global_k": ['1', '3','10','30', '100', '300', '1000']}

for lr in vi_lrs:
    Ks[f"vi_{lr}"] = vi_iter_counts

plotAllForSinglePlateCombination(Ks, {"experimentName": "bus_breakdown", "plateSizes": {"M": 3, "J": 3, "I": 30}}, "alpha", vi_lrs=vi_lrs)

plotAllForSinglePlateCombination(Ks, {"experimentName": "bus_breakdown", "plateSizes": {"M": 3, "J": 3, "I": 30}}, "alpha", vi_lrs=[])

