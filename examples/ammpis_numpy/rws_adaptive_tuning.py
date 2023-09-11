from ammpis import *
import matplotlib.pyplot as plt
import torch as t
from torch.distributions import Normal, Uniform
import pickle
from itertools import product
import math

matplotlib.pyplot.rc('text', usetex=True)

def x_axis(xs, use_time=False):
    if use_time:
        return xs
    else:
        return range(len(xs))

T = 500

constant_lrs = [0.01]#, 0.1]
num_constant_lr = len(constant_lrs)
lr_funcs = []
for lr in constant_lrs:
    # lr_funcs.append(lambda i, p: lr)
    lr_funcs.append(lr)


lr_funcs += [lambda i, p: 1 / ((i+10)**p)]#,
            #  lambda i, p: 1 / ((i+100)**p),
            #  lambda i, p: 1 / ((i+1000)**p)]#,
            #  lambda i, p: min(0.1, 1 / ((i+1)**p)),
            #  lambda i, p: min(0.3, 1 / ((i+1)**p)),
            #  lambda i: 0.1 if i < 250 else 0.01,
            #  lambda i: 0.1 if i < 250 else 0.01 if i < 1000 else 0.001]#,
            #  lambda i: 0.1 if i < 250 else 0.01 if i < 500 else 0.001 if i < 750 else 0.01 if i < 1000 else 0.001]#,
            #  lambda i, p: max(min(0.3, p**math.floor(i/(T/100))), 0.0001)]
pows = [0.75, 0.9]

Ns = [500]#50, 500, 5000]
Ks = [5]#, 10, 30, 100]
var_sizes = ["WIDE"]#, "NARROW"]
# var_sizes = ["NARROW"]

total_num_plots = len(Ns) * len(Ks) * len(var_sizes)

plot_counter = 0
for num_latents, K, VAR_SIZE_STR in product(Ns, Ks, var_sizes):
    plot_counter += 1
    
    t.manual_seed(1)
    t.cuda.manual_seed(0)

    print(f"({plot_counter}/{total_num_plots}) Running for N={num_latents}, K={K}, VAR_SIZE={VAR_SIZE_STR}")

    init = t.tensor([0.0,1.0], dtype=t.float64).repeat((num_latents,1))

    loc = Normal(0,1).sample((num_latents,1)).float()
    if VAR_SIZE_STR == "WIDE":
        scale = Normal(0,1).sample((num_latents,1)).exp().float()
    else:
        scale = Uniform(-6,-1).sample((num_latents,1)).exp().float()

    print(f'Location mean: {loc.abs().mean()}')
    print(f'Variance mean: {scale.mean()}')
    post_params = t.cat([loc, scale], dim=1)

    # first get the results
    rws_variants = [natural_rws]#, natural_rws_standardised]

    # natural_rws and natural_rws_standardised
    rws_results = {}
    for i, fn in enumerate(rws_variants):
        for j, lr_fn in enumerate(lr_funcs):
            for k, p in enumerate(pows):
                if not(j < num_constant_lr and k > 0):
                    if j < num_constant_lr:
                        lr = lr_fn
                    else:
                        if lr_fn.__code__.co_argcount == 1:
                            if k > 0:
                                continue
                            lr = lr_fn
                        else:
                            lr = lambda i: lr_fn(i, p)
                    # breakpoint()
                    # rws_results[(i,j,k)] = fn(T, post_params, init, lr, K)
                    # rws_results[(i,j,k)] = fn(T, post_params, init, lr, K)
                    rws_results[(i,j,k)] = fn(T, init, lr, K, post_params=post_params, post_type=Normal)
                    print(f"{fn.__name__}, lr_type={j}, p={p} done.")
            
    # rws_results['natural_rws_difference'] = natural_rws_difference(T, post_params, init, 0.005, K)
    # print(f"natural_rws_difference, lr=0.005 done.")

    # HMC
    hmc_params   = {"N": num_latents,
                    "T": T//4,
                    "post_params": post_params,
                    "init": init,
                    "post_type": Normal,
                    "num_chains": 4}

    with open('saved_hmc.pkl', 'rb') as f:
        saved_hmc = pickle.load(f)

    params_match = False
    if all([hmc_params[key] == saved_hmc["params"][key] for key in ["N", "T", "post_type", "num_chains"]]):
        if all([(hmc_params[key] == saved_hmc["params"][key]).all() for key in ["post_params", "init"]]):
            print("Loading saved HMC results.")
            hmc_moms, hmc_times, hmc_samples = saved_hmc['results']
            params_match = True

    if not params_match:
        n = hmc_params.pop("N")
        hmc_moms, hmc_times, hmc_samples = HMC(**hmc_params)
        hmc_params["N"] = n
        with open('saved_hmc.pkl', 'wb') as f:
            pickle.dump({'params': hmc_params, 'results': (hmc_moms, hmc_times, hmc_samples)}, f)
        
    print(f"HMC done.")

    # ammp_is_m_q, ammp_is_m_avg, ammp_is_l_tot, ammp_is_l_one_iters, ammp_is_log_weights, ammp_is_entropies, ammp_is_times = ammp_is(T, post_params, init, 0.03, K)
    # print(f"ammp-is done.")
   
    vi_means, vi_vars, vi_elbos, vi_entropies, vi_times = VI(T, post_params, init, 0.05, K=K)
    print(f"VI done.")

    m_mcmc, mcmc_acceptance_rate, mcmc_times, mcmc_samples = mcmc(T, post_params, init, 2.4*scale, burn_in=T//10)
    print(f"MCMC done.")
    
    m_lang, lang_acceptance_rate, lang_times, lang_samples = lang(T, post_params, init, 1e-6*scale, burn_in=T//10)        
    print(f"Lang done.")


    for use_time in [True, False]:

        colours = ['r', 'g', 'b', 'c', 'm', 'y','orange', 'purple', 'brown', 'pink', 'grey']
        # colours = ['#543005','#8c510a','#bf812d','#dfc27d', "#80cdc1", "#c7eae5", "#f6e8c3", "#FF69B4", "pink"]
        # colours = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6']

        fig, ax = plt.subplots(5,1, figsize=(5.5, 14.0))

        final_lines = []
        colour_count = 0

        for i, fn in enumerate(rws_variants):
            for j, lr_fn in enumerate(lr_funcs):
                for k, p in enumerate(pows):
                    if not(j < num_constant_lr and k > 0):
                        if j < num_constant_lr:
                            p_str = f", lr={constant_lrs[j]}"
                        else:
                            if lr_fn.__code__.co_argcount == 1:
                                if k > 0:
                                    continue
                                p_str = f"{j - num_constant_lr + 1}"
                            else:
                                p_str = f"{j - num_constant_lr + 1}, p={p}"

                        m_q, l_one_iters, entropies, times = rws_results[(i,j,k)]
                        mean_errs, var_errs = get_errs(m_q, post_params)

                        l0, = ax[0].plot(x_axis(times, use_time), mean_errs, c=colours[colour_count], alpha = 1-k/len(pows), label=f'{fn.__name__[8:]}{p_str}')
                        l1, = ax[1].plot(x_axis(times, use_time), var_errs, c=colours[colour_count], alpha = 1-k/len(pows), label=f'{fn.__name__[8:]}{p_str}')
                        l2, = ax[2].plot(x_axis(times[3:], use_time), [lt for lt in l_one_iters][2:], c=colours[colour_count], alpha = 1-k/len(pows), label=f'{fn.__name__[8:]}{p_str}')
                        l3, = ax[3].plot(x_axis(times, use_time), entropies, c=colours[colour_count], alpha = 1-k/len(pows), label=f'{fn.__name__[8:]}{p_str}')
                        l4, = ax[4].plot(x_axis(times, use_time), [(fit_approx_post(m)[:,1] / post_params[:,1]).mean() for m in m_q], c=colours[colour_count], alpha = 1-k/len(pows), label=f'{fn.__name__[8:]}{p_str}')

                        if k == 0 or True:
                            final_lines.append(l0)

                colour_count += 1

        # m_q, l_one_iters, entropies, times = rws_results['natural_rws_difference']
        # mean_errs, var_errs = get_errs(m_q, post_params)

        # l0, = ax[0].plot(x_axis(times, use_time), mean_errs, c=colours[colour_count], label=f'natural_rws_difference'[8:])
        # l1, = ax[1].plot(x_axis(times, use_time), var_errs, c=colours[colour_count], label=f'natural_rws_difference'[8:])
        # l2, = ax[2].plot(x_axis(times[3:], use_time), [lt for lt in l_one_iters][2:], c=colours[colour_count], label=f'natural_rws_difference'[8:])
        # l3, = ax[3].plot(x_axis(times, use_time), entropies, c=colours[colour_count], label=f'natural_rws_difference'[8:])
        # l4, = ax[4].plot(x_axis(times, use_time), [(fit_approx_post(m)[:,1] / post_params[:,1]).mean() for m in m_q], c=colours[colour_count], label=f'natural_rws_difference'[8:])

        # final_lines.append(l0)
        
        # mean_errs, var_errs = get_errs(ammp_is_m_q, post_params)

        # ax[0].plot(x_axis(ammp_is_times, use_time), mean_errs, c=colours[colour_count], label=f'ammp-is')
        # ax[1].plot(x_axis(ammp_is_times, use_time), var_errs, c=colours[colour_count], label=f'ammp-is')
        # ax[2].plot(x_axis(ammp_is_times[3:], use_time), [lt + lw for lt, lw in zip(ammp_is_l_one_iters, ammp_is_log_weights)][2:], c=colours[colour_count], label=f'ammp-is')
        # ax[3].plot(x_axis(ammp_is_times, use_time), ammp_is_entropies, c=colours[colour_count], label=f'ammp-is')
        # ax[4].plot(x_axis(ammp_is_times, use_time), [(fit_approx_post(m)[:,1] / post_params[:,1]).mean() for m in ammp_is_m_avg], c=colours[colour_count], label=f'ammp-is')

        # colour_count += 1

        # HMC
        hmc_mean_errs, hmc_var_errs = get_errs(hmc_moms, post_params)

        hmc_line, = ax[0].plot(x_axis(hmc_times, use_time)[:len(hmc_mean_errs)], hmc_mean_errs, c='black', label='HMC')
        final_lines.append(hmc_line)
        ax[1].plot(x_axis(hmc_times, use_time)[:len(hmc_var_errs)], hmc_var_errs, c='black', label='HMC')

        # VI
        mean_errs = []
        var_errs = []

        for i in range(len(vi_means)):
            mean_errs.append((post_params[:,0] - vi_means[i]).abs().mean())
            var_errs.append((post_params[:,1] - vi_vars[i].exp()).abs().mean())

        ax[0].plot(x_axis(vi_times[:-1], use_time), mean_errs, c='#54278f', label='VI')
        ax[1].plot(x_axis(vi_times[:-1], use_time), var_errs, c='#54278f', label='VI')
        ax[2].plot(x_axis(vi_times[:-1], use_time), vi_elbos, c='#54278f', label='VI') 
        ax[3].plot(x_axis(vi_times[:-1], use_time), vi_entropies, c='#54278f', label='VI')
        ax[4].plot(x_axis(vi_times[:-1], use_time), [(v.exp() / post_params[:,1]).mean() for v in vi_vars], c='#54278f', label='VI')


        # MCMC
        colour_count += 2
        mean_errs_mcmc, var_errs_mcmc = get_errs(m_mcmc, post_params)

        ax[0].plot(x_axis(mcmc_times, use_time), mean_errs_mcmc, c='b', label='MCMC')
        ax[1].plot(x_axis(mcmc_times, use_time), var_errs_mcmc, c='b', label='MCMC')

        # Lang
        mean_errs_lang, var_errs_lang = get_errs(m_lang, post_params)

        ax[0].plot(x_axis(lang_times, use_time), mean_errs_lang, c='brown', label='Lang')
        ax[1].plot(x_axis(lang_times, use_time), var_errs_lang, c='brown', label='Lang')


        ax[0].set_title("Mean Error")
        ax[1].set_title("Var Error")
        ax[2].set_title("weighted l_one_iter")
        ax[3].set_title("Entropy")
        ax[4].set_title("Q variance / posterior variance")

        # log y axis
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        # ax[2].set_yscale('symlog')
        # ax[3].set_yscale('symlog')

        ax[2].set_ylim(-50, 20)
        
        ax[4].set_ylim(0.5, 1.5)
        # ax[4].set_ylim(0,20)


        ax[0].legend(loc='upper right')

        ax[-1].set_xlabel('Time (s)' if use_time else 'Iteration')
        if use_time:
            for a in ax:
                a.set_xscale('log')
                a.set_xlim(-0.001)

        # ax[0].legend(handles = final_lines, loc='lower center', bbox_to_anchor=(0.5, 1.25), shadow=False, ncol=2)
        ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.25), shadow=False, ncol=3)

        # plt.suptitle("Natural RWS Learning Rate Schedules\n1: $(i+10)^{-p}$\n2: min($0.1, (i+1)^{-p}$)\n3: min($0.3, (i+1)^{-p}$)\n4: $0.1$ for $i < 250$, $0.01$ for $250 < i < 1000$, $0.001$ for $i > 1000$")
        # plt.suptitle("Natural RWS Learning Rate Schedules\n1: $(i+10)^{-p}$")
        plt.suptitle("Natural RWS Learning Rate Schedules\n1: $(i+10)^{-p}$\n2: $(i+100)^{-p}$\n3: $(i+1000)^{-p}$")

        plt.tight_layout()  
        plt.savefig(f"figures/rws_lrs_adaptive/N{num_latents}_K{K}_{VAR_SIZE_STR}{'_TIME' if use_time else ''}_NEWER.png")
        plt.savefig(f"figures/rws_lrs_adaptive/N{num_latents}_K{K}_{VAR_SIZE_STR}{'_TIME' if use_time else ''}_NEWER.pdf")

        # if use_time:
        #     for a in ax:
        #         a.set_xlim()
        #         a.set_xscale('log')

        # plt.savefig(f"figures/rws_lrs_adaptive/N{num_latents}_K{K}_{VAR_SIZE_STR}{'_logTIME' if use_time else ''}.png")
        # plt.savefig(f"figures/rws_lrs_adaptive/N{num_latents}_K{K}_{VAR_SIZE_STR}{'_logTIME' if use_time else ''}.pdf")


        plt.close()

    print()