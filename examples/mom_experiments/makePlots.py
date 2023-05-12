import json
import matplotlib.pyplot as plt
from tueplots import axes, bundles
import combineJSONs
import numpy as np

def combineVIJsons(vi_lrs, experimentName, eval_k=1, dataset_seed=0):
    for metric in ["elbo", "MSE", "p_ll", "variance"]:
        combineJSONs.combineMultiple(
            [f"{experimentName}/results/vi_{experimentName}_{metric}_lr{lr}_{dataset_seed}.json" for lr in vi_lrs],
             f"{experimentName}/results/vi_{experimentName}_{metric}_{dataset_seed}.json", sublayer=str(eval_k))
            
def combineALLJsons(experimentName, NUTS=False, dataset_seed=0):
    fs = [f"{experimentName}/results/{experimentName}_elbo_{dataset_seed}.json",
          f"{experimentName}/results/vi_{experimentName}_elbo_{dataset_seed}.json"]

    combineJSONs.combineMultiple(fs, f"{experimentName}/results/ALL_{experimentName}_elbo_{dataset_seed}.json")    

    for metric in ["MSE", "p_ll", "variance"]:
        fs = [f"{experimentName}/results/{experimentName}_{metric}_{dataset_seed}.json",
          f"{experimentName}/results/vi_{experimentName}_{metric}_{dataset_seed}.json"]
        if NUTS:
            fs.append(f"{experimentName}/results/hmc_{experimentName}_{metric}_{dataset_seed}.json")

        combineJSONs.combineMultiple(fs, f"{experimentName}/results/ALL_{experimentName}_{metric}_{dataset_seed}.json")
        
def averageALLResultsOverDatasets(experimentName, dataset_seeds):
    n = len(dataset_seeds)

    for metric in ["elbo", "MSE", "p_ll", "variance"]:
        fs = [f"{experimentName}/results/ALL_{experimentName}_{metric}_{dataset_seed}.json" for dataset_seed in dataset_seeds]

        js = []
        for f_in in fs:
            with open(f_in) as f:
                js.append(json.load(f))

        j_avg = js[0]
        for method in j_avg:
            for K in j_avg[method]:
                j_avg[method][K]["time_mean"] = sum([js[i][method][K]["time_mean"] for i in range(n)])/n
                j_avg[method][K]["time_std_err"] = sum([js[i][method][K]["time_mean"] for i in range(n)])/n

                try:
                    j_avg[method][K]["mean"] = sum([js[i][method][K]["mean"] for i in range(n)])/n
                    j_avg[method][K]["std_err"] = sum([js[i][method][K]["std_err"] for i in range(n)])/n

                except KeyError:
                    for rv in j_avg[method][K]:
                        if rv not in ("time_mean", "time_std_err"):
                            j_avg[method][K][rv]["mean_var"] = sum([js[i][method][K][rv]["mean_var"] for i in range(n)])/n

        with open(f"{experimentName}/results/ALL_{experimentName}_{metric}.json", 'w') as f:
            json.dump(j_avg, f, indent=4)


def getSeriesValues(data, method, Ks, valueType, rv=None, rolling_window=1):
    if rolling_window == 1:
        if rv is None:
            return [data[method][k][valueType] for k in Ks[method]]
        else:
            return [data[method][k][rv][valueType] for k in Ks[method]]
    else:
        vals = []
        for i, k in enumerate(Ks[method]):
            valsToAverage = []
            for j in range(rolling_window):
                if i-j >= 0:
                    if rv is None:
                        valsToAverage.append(data[method][Ks[method][i-j]][valueType])
                    else:
                        valsToAverage.append(data[method][Ks[method][i-j]][rv][valueType])
            vals.append(sum(valsToAverage)/len(valsToAverage))   
        return vals 
    
def getMeanValues(data, method, Ks, rv=None, rolling_window=1):
    # print(data, method, Ks)
    if rv is None:
        return getSeriesValues(data, method, Ks, "mean", rolling_window=rolling_window)
    else:
        return getSeriesValues(data, method, Ks, "mean_var", rv, rolling_window=rolling_window)
    
def getStdErrValues(data, method, Ks, rv=None, rolling_window=1):
    if rv is None:
        return [x*(1000**0.5) for x in getSeriesValues(data, method, Ks, "std_err", rolling_window=rolling_window)]
    else:
        return []

def getTimeValues(data, method, Ks, rv=None, rolling_window=1):
    if rv is None:
        return getSeriesValues(data, method, Ks, "time_mean", rolling_window=rolling_window)
    else:
        return getSeriesValues(data, method, Ks, "time_mean", rolling_window=rolling_window)#, rv)
    

greenColourRGB = (77/256, 175/256, 74/256)

def plotAllForSinglePlateCombination(Ks, experimentName, rv, vi_lrs=[0.01,0.001,0.0001], vi_rolling_window=1, vi_eval_k=1, NUTS=False, dataset_seeds=[0]):
    
    # Combine different baselines results into singular "ALL" files per dataset_seed
    for dataset_seed in dataset_seeds:
        if len(vi_lrs) >= 1:
            combineVIJsons(vi_lrs, experimentName, str(vi_eval_k), dataset_seed=dataset_seed)
            combineALLJsons(experimentName, NUTS, dataset_seed)

    # Average over all dataset_seeds to obtain a master "ALL" file (with no "_{dataset_seed}" at end of filename)
    averageALLResultsOverDatasets(experimentName, dataset_seeds)


    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context(bundles.icml2022()):
        
        with open(f'{experimentName}/results/ALL_{experimentName}_elbo.json') as f:
            elbos = json.load(f)
        with open(f'{experimentName}/results/ALL_{experimentName}_p_ll.json') as f:
            p_lls = json.load(f)
        with open(f'{experimentName}/results/ALL_{experimentName}_variance.json') as f:
            vars_ = json.load(f)
        with open(f'{experimentName}/results/ALL_{experimentName}_MSE.json') as f:
            mses = json.load(f)

        results = {}
        metricToData = {"elbo": elbos, "MSE": mses, "p_ll": p_lls, "vars": vars_}
        for metric in metricToData: 
            data = metricToData[metric]
            results[metric] = {}
            for method in Ks:
                if not (method == "NUTS" and (metric == "elbo" or not NUTS)):
                    if metric in ("elbo", "p_ll"):
                        rv_temp = None
                    else:
                        rv_temp = rv
                    if "vi" in method and vi_rolling_window > 1:
                        results[metric][method] = {"mean": getMeanValues(data, method, Ks, rv_temp, vi_rolling_window),
                                                "std_err": getStdErrValues(data, method, Ks, rv_temp, vi_rolling_window)*np.sqrt(1000),
                                                "time": getTimeValues(data, method, Ks, rv_temp, vi_rolling_window)}
                    else:
                        results[metric][method] = {"mean": getMeanValues(data, method, Ks, rv_temp),
                                                "std_err": getStdErrValues(data, method, Ks, rv_temp),
                                                "time": getTimeValues(data, method, Ks, rv_temp)}

        # if NUTS:
        #     with open(f'{experimentName}/results/hmc_{experimentName}TEST_p_ll.json') as f:
        #         nuts_p_ll = json.load(f)
        #     results["p_ll"]["NUTS"] = {"mean": getMeanValues(nuts_p_ll, "NUTS", Ks),
        #                                 "std_err": getStdErrValues(nuts_p_ll, "NUTS", Ks),
        #                                 "time": getTimeValues(nuts_p_ll, "NUTS", Ks)}
        #     with open(f'{experimentName}/results/hmc_{experimentName}TEST_variance.json') as f:
        #         nuts_vars = json.load(f)
        #     results["vars"]["NUTS"] = {"mean": getMeanValues(nuts_vars, "NUTS", Ks, rv),
        #                                 "std_err": getStdErrValues(nuts_vars, "NUTS", Ks, rv),
        #                                 "time": getTimeValues(nuts_vars, "NUTS", Ks, rv)}


        fig, ax = plt.subplots(2,4,figsize=(8.5, 4.5))
        
        ax[0,0].errorbar(Ks["tmc_new"],  results["elbo"]["tmc_new"]["mean"],  yerr=results["elbo"]["tmc_new"]["std_err"],  linewidth=0.55, markersize = 0.75, fmt='-o', c='#e41a1c', label='MP IS')
        ax[0,0].errorbar(Ks["global_k"], results["elbo"]["global_k"]["mean"], yerr=results["elbo"]["global_k"]["std_err"], linewidth=0.55, markersize = 0.75, fmt='-o', c='#377eb8', label='Global IS')

        ax[0,1].errorbar(Ks["tmc_new"][1:],  results["p_ll"]["tmc_new"]["mean"][1:],  yerr=results["p_ll"]["tmc_new"]["std_err"][1:],  linewidth=0.55, markersize = 0.75, fmt='-o', c='#e41a1c', label='MP IS')
        ax[0,1].errorbar(Ks["global_k"][1:], results["p_ll"]["global_k"]["mean"][1:], yerr=results["p_ll"]["global_k"]["std_err"][1:], linewidth=0.55, markersize = 0.75, fmt='-o', c='#377eb8', label='Global IS')

        ax[0,2].errorbar(Ks["tmc_new"][1:],  results["vars"]["tmc_new"]["mean"][1:],  yerr=0,  linewidth=0.55, markersize = 0.75, fmt='-o', c='#e41a1c', label='MP IS')
        ax[0,2].errorbar(Ks["global_k"][1:], results["vars"]["global_k"]["mean"][1:], yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='#377eb8', label='Global IS')

        ax[0,3].errorbar(Ks["tmc_new"][1:],  results["MSE"]["tmc_new"]["mean"][1:],  yerr=0,  linewidth=0.55, markersize = 0.75, fmt='-o', c='#e41a1c', label='MP IS')
        ax[0,3].errorbar(Ks["global_k"][1:], results["MSE"]["global_k"]["mean"][1:], yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='#377eb8', label='Global IS')


        ax[1,0].errorbar(results["elbo"]["tmc_new"]["time"],  results["elbo"]["tmc_new"]["mean"],  yerr=0,  linewidth=0.55, markersize = 0.75, fmt='-o', c='#e41a1c', label='MP IS')
        ax[1,0].errorbar(results["elbo"]["global_k"]["time"], results["elbo"]["global_k"]["mean"], yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='#377eb8', label='Global IS')
        for alpha, lr in enumerate(vi_lrs[::-1]):
            # if not(experimentName == "bus_breakdown" and lr in [0.1, 0.01]):
            ax[1,0].errorbar(results["elbo"][f"vi_{lr}"]["time"][:101], results["elbo"][f"vi_{lr}"]["mean"][:101], yerr=0, linewidth=0.55, markersize = 0.75, label="VI lr="+str(lr), c=(*greenColourRGB, 1/(alpha+1)))

        # ax[1,0].set_yscale("log")
        if experimentName == "bus_breakdown":
            ax[1,0].set_ylim(-40000, 0)

        ax[1,1].errorbar(results["p_ll"]["tmc_new"]["time"][1:],  results["p_ll"]["tmc_new"]["mean"][1:],  yerr=0,  linewidth=0.55, markersize = 0.75, fmt='-o', c='#e41a1c', label='MP IS')
        ax[1,1].errorbar(results["p_ll"]["global_k"]["time"][1:], results["p_ll"]["global_k"]["mean"][1:], yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='#377eb8', label='Global IS')
        for alpha, lr in enumerate(vi_lrs[::-1]):
            ax[1,1].errorbar(results["p_ll"][f"vi_{lr}"]["time"][:101], results["p_ll"][f"vi_{lr}"]["mean"][:101], yerr=0, linewidth=0.55, markersize = 0.75, label="VI lr="+str(lr), c=(*greenColourRGB, 1/(alpha+1)))
        if NUTS:
            ax[1,1].errorbar(results["p_ll"]["NUTS"]["time"], results["p_ll"]["NUTS"]["mean"], yerr=0, linewidth=0.55, markersize = 0.75, label="NUTS", c='#984ea3')

        ax[1,2].errorbar(results["vars"]["tmc_new"]["time"][1:],  results["vars"]["tmc_new"]["mean"][1:],  yerr=0,  linewidth=0.55, markersize = 0.75, fmt='-o', c='#e41a1c', label='MP IS')
        ax[1,2].errorbar(results["vars"]["global_k"]["time"][1:], results["vars"]["global_k"]["mean"][1:], yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='#377eb8', label='Global IS')
        for alpha, lr in enumerate(vi_lrs[::-1]):
            ax[1,2].plot(results["vars"][f"vi_{lr}"]["time"][:101], results["vars"][f"vi_{lr}"]["mean"][:101], linewidth=0.55, markersize = 0.75, label="VI lr="+str(lr), c=(*greenColourRGB, 1/(alpha+1)))
        if NUTS:
            ax[1,2].plot(results["vars"]["NUTS"]["time"], results["vars"]["NUTS"]["mean"], linewidth=0.55, markersize = 0.75, label="NUTS", c='#984ea3')

        ax[1,3].errorbar(results["MSE"]["tmc_new"]["time"][1:],  results["MSE"]["tmc_new"]["mean"][1:],  yerr=0,  linewidth=0.55, markersize = 0.75, fmt='-o', c='#e41a1c', label='MP IS')
        ax[1,3].errorbar(results["MSE"]["global_k"]["time"][1:], results["MSE"]["global_k"]["mean"][1:], yerr=0, linewidth=0.55, markersize = 0.75, fmt='-o', c='#377eb8', label='Global IS')
        for alpha, lr in enumerate(vi_lrs[::-1]):
            ax[1,3].plot(results["MSE"][f"vi_{lr}"]["time"], results["MSE"][f"vi_{lr}"]["mean"], linewidth=0.55, markersize = 0.75, label="VI lr="+str(lr), c=(*greenColourRGB, 1/(alpha+1)))
        if NUTS:
            ax[1,3].plot(results["MSE"]["NUTS"]["time"], results["MSE"]["NUTS"]["mean"], linewidth=0.55, markersize = 0.75, label="NUTS", c='#984ea3')

        if rv == "alpha": rv = "BoroughMean"  # change to match the name of the variable in the paper

        for j in range(4):
            colTitles = ['a','b','c','d']
            ax[0,j].set_xlabel('K')
            ax[1,j].set_xlabel('Time (s)')

            ax[0,j].tick_params(axis='x', labelrotation = 90)

            ax[1,j].set_xlim(left=-0.01 if experimentName == "movielens" else -0.05)
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
            ax[i,2].set_ylabel(f"Variance of \n{rv} Estimator", fontsize=10)
            ax[i,3].set_ylabel(f"MSE of \n{rv} Estimator", fontsize=10)

        plt.legend(loc="center right")
        plt.savefig(f'plots/{experimentName}_all_{rv}.png')
        plt.savefig(f'plots/{experimentName}_all_{rv}.pdf')
        plt.close()


# MOVIELENS
vi_lrs= [0.0001, 0.001, 0.01]

num_vi_iters = 250
vi_iter_step = 1
vi_iter_counts = [str(x) for x in range(0, num_vi_iters+1, vi_iter_step)]

num_NUTS_samples = 6

Ks = {"tmc_new": ['1', '3', '10', '30', '100'], "global_k": ['1', '3','10','30', '100', '300', '1000', '3000', '10000', '30000', '100000']}

for lr in vi_lrs:
    Ks[f"vi_{lr}"] = vi_iter_counts

Ks["NUTS"] = [str(x) for x in range(1, num_NUTS_samples+1)]

plotAllForSinglePlateCombination(Ks, "movielens", "z", vi_lrs=vi_lrs, NUTS=True)

# BUS BREAKDOWN
vi_lrs=[0.0001, 0.001, 0.01]

num_vi_iters = 100#200
vi_iter_step = 1
vi_iter_counts = [str(x) for x in range(0, num_vi_iters+1, vi_iter_step)]

num_NUTS_samples = 5

Ks = {"tmc_new": ['1', '3', '10', '30', '100'], "global_k": ['1', '3','10','30', '100', '300', '1000', '3000', '10000', '30000', '100000']}

for lr in vi_lrs:
    Ks[f"vi_{lr}"] = vi_iter_counts

Ks["NUTS"] = [str(x) for x in range(1, num_NUTS_samples+1)]

plotAllForSinglePlateCombination(Ks, "bus_breakdown", "alpha", vi_lrs=vi_lrs, NUTS=True)

