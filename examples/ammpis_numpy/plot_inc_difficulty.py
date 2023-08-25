from ammpis import *
import matplotlib.pyplot as plt
import torch as t
from torch.distributions import Normal, Uniform
import pickle
import sys
from get_posteriors import get_posteriors

posteriors = get_posteriors()

to_plot = []

args = sys.argv[1:]
if len(args) == 0:
    to_plot = [0,1,2,3,4]
else:
    for i, x in enumerate(args):
        to_plot.append(int(x))

rws_names = ["rws, lr=0.01", "rws*, p=0.75", "rws*, p=0.9"]

ammp_is_variants = []#ammp_is]

results_collection = {}
post_params_collection = {}

for j in to_plot:
    posterior_settings = posteriors[j]

    with open(f"saved_results/increasing_difficulty{j}.pkl", "rb") as f:
        data = pickle.load(f)

    results = data["results"]
    post_params = data["post_params"]

    results_collection[j] = results
    post_params_collection[j] = post_params

    print(f"Posterior {j} results loaded.")

# fig, ax = plt.subplots(4, len(posteriors), figsize=(28.5, 14.0))
fig, ax = plt.subplots(4, len(to_plot), figsize=(16.5, 7.5))

for j, post_idx in enumerate(to_plot):
    posterior_settings = posteriors[post_idx]

    results = results_collection[post_idx]
    post_params = post_params_collection[post_idx]

    num_latents  = posterior_settings["N"]
    K            = posterior_settings["K"]
    VAR_SIZE_STR = posterior_settings["VAR_SIZE"]
    LOC_VAR   = posterior_settings["LOC_VAR"]

    # # AMMPIS
    # colours = ['#543005','#8c510a','#bf812d','#dfc27d', "pink", "#FF69B4", "#f6e8c3", "#c7eae5", "#80cdc1"]
    colours = ['orange', 'g', 'b', 'y', 'r']
    for i, fn in enumerate(ammp_is_variants):
        mean_errs, var_errs, ammp_is_times = results[fn.__name__]

        ax[0, j].plot(ammp_is_times, mean_errs, c=colours[i], label=f'{fn.__name__}')
        ax[1, j].plot(ammp_is_times, mean_errs, c=colours[i], label=f'{fn.__name__}')
        ax[2, j].plot(ammp_is_times, var_errs, c=colours[i], label=f'{fn.__name__}')
        ax[3, j].plot(ammp_is_times, var_errs, c=colours[i], label=f'{fn.__name__}')

    # NATURAL RWS
    # rws_colours = ['#80cdc1','#35978f','#01665e']
    rws_colours = ['r', '#66c2a4', '#006d2c']

    for i, rws_name in enumerate(rws_names):
        mean_errs, var_errs, times = results[f"rws{i}"] 

        if post_idx == 5 and rws_name == "rws, lr=0.01":
            num_samples = len(mean_errs)
            mean_errs = mean_errs[:20000]
            var_errs = var_errs[:20000]
            times = times[:20000]

        ax[0, j].plot(times, mean_errs, c=rws_colours[i], label=f'{rws_name}')
        ax[1, j].plot(times, mean_errs, c=rws_colours[i], label=f'{rws_name}')

        ax[2, j].plot(times, var_errs, c=rws_colours[i], label=f'{rws_name}')
        ax[3, j].plot(times, var_errs, c=rws_colours[i], label=f'{rws_name}')

    # HMC
    hmc_mean_errs, hmc_var_errs, hmc_times = results["HMC"]

    ax[0, j].plot(hmc_times[:len(hmc_mean_errs)], hmc_mean_errs, c='black', label='HMC')
    ax[1, j].plot(hmc_times[:len(hmc_mean_errs)], hmc_mean_errs, c='black', label='HMC')
    ax[2, j].plot(hmc_times[:len(hmc_var_errs)], hmc_var_errs, c='black', label='HMC')
    ax[3, j].plot(hmc_times[:len(hmc_var_errs)], hmc_var_errs, c='black', label='HMC')


    # VI
    mean_errs, var_errs, vi_times = results["VI"]

    if post_idx == 13:
        vi_num_samples = len(mean_errs)
        mean_errs = mean_errs[:vi_num_samples//3]
        var_errs = var_errs[:vi_num_samples//3]
        vi_times = vi_times[:1+vi_num_samples//3]

    ax[0, j].plot(vi_times[:-1], mean_errs, c='#54278f', label='VI')
    ax[1, j].plot(vi_times[:-1], mean_errs, c='#54278f', label='VI')
    ax[2, j].plot(vi_times[:-1], var_errs, c='#54278f', label='VI')
    ax[3, j].plot(vi_times[:-1], var_errs, c='#54278f', label='VI')

    # MCMC
    mean_errs_mcmc, var_errs_mcmc, mcmc_times = results["mcmc"]

    if post_idx == 13:
        mcmc_num_samples = len(mean_errs_mcmc)
        mean_errs_mcmc = mean_errs_mcmc[:mcmc_num_samples//3]
        var_errs_mcmc = var_errs_mcmc[:mcmc_num_samples//3]
        mcmc_times = mcmc_times[:mcmc_num_samples//3]

    ax[0, j].plot(mcmc_times, mean_errs_mcmc, c='b', label='MCMC')
    ax[1, j].plot(mcmc_times, mean_errs_mcmc, c='b', label='MCMC')
    ax[2, j].plot(mcmc_times, var_errs_mcmc, c='b', label='MCMC')
    ax[3, j].plot(mcmc_times, var_errs_mcmc, c='b', label='MCMC')

    # LANG
    # mean_errs_lang, var_errs_lang, lang_times = results["lang"]
    # # breakpoint()

    # ax[0, j].plot(lang_times, mean_errs_lang, c='brown', label='Lang')
    # ax[1, j].plot(lang_times, mean_errs_lang, c='brown', label='Lang')
    # ax[2, j].plot(lang_times, var_errs_lang, c='brown', label='Lang')
    # ax[3, j].plot(lang_times, var_errs_lang, c='brown', label='Lang')


    # Formatting
    # ax[0, j].set_title(f"Mean Error, N={num_latents}")
    # ax[2, j].set_title("Var Error")
    ax[0,j].set_title(f"N={num_latents}, K={K},\n{VAR_SIZE_STR} posterior,\n{'LARGE' if LOC_VAR > 1 else 'SMALL'} location var")
    # ax[0,j].set_title(f"({['a', 'b', 'c', 'd', 'e'][j]})")

    for k in range(4):
        # log y axis
        ax[k, j].set_yscale('log')

        # if k % 2 == 0:
        #     ax[k, j].set_xlim(-0.1, hmc_times[-1]*1.1)
        # else:
        #     ax[k, j].set_xlim(1e-2, hmc_times[-1]*1.1)

        # log x axis every other row
        if k % 2 == 1:
            ax[k,j].set_xscale('log')

    ax[-1, j].set_xlabel('Time (s)')
    

# ax[0, 4].legend(loc='upper center', bbox_to_anchor=(1.4, 0.1), shadow=False, ncol=1)
ax[0, 0].legend(loc='lower center', bbox_to_anchor=(2.85, 1.46), shadow=False, ncol=7)

plt.tight_layout()

ax[1,0].text(-0.4, 0.7, 'Mean Error', va='bottom', rotation='vertical', transform=ax[1,0].transAxes, size="xx-large")
ax[3,0].text(-0.4, -1.6, 'Var Error',  va='bottom', rotation='vertical', transform=ax[1,0].transAxes, size="xx-large")

# plt.tight_layout()  
plt.savefig(f"figures/difficulty_plots/increasing_difficulty{to_plot}.png")
plt.savefig(f"figures/difficulty_plots/increasing_difficulty{to_plot}.pdf")
plt.close()
