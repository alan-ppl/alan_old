from ammpis import *
import matplotlib.pyplot as plt
import torch as t
from torch.distributions import Normal, Uniform
import pickle
from itertools import product

def x_axis(xs, use_time=False):
    if use_time:
        return xs
    else:
        return range(len(xs))

T = 2000
lrs = [0.01, 0.03, 0.1, 0.3]

Ns = [50, 500, 5000]
Ks = [3, 10, 30, 100]
var_sizes = ["WIDE", "NARROW"]

total_num_plots = len(Ns) * len(Ks) * len(var_sizes)

plot_counter = 0
for num_latents, K, VAR_SIZE_STR in product(Ns, Ks, var_sizes):
    plot_counter += 1

    if plot_counter <= 4:
        continue

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
    ammp_is_variants = [ammp_is, ammp_is_ReLU, ammp_is_uniform_dt, ammp_is_no_inner_loop, ammp_is_no_inner_loop_ReLU]#, ammp_is_weight_all]

    ammp_is_results = {}
    for i, fn in enumerate(ammp_is_variants):
        for j, lr in enumerate(lrs):
            ammp_is_results[(i,j)] = fn(T, post_params, init, lr, K)
            print(f"{fn.__name__}, lr={lr} done.")

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


    for use_time in [True, False]:

        colors = ['r', 'g', 'b', 'c', 'm', 'y']

        fig, ax = plt.subplots(5,1, figsize=(5.5, 14.0))

        final_lines = []

        for i, fn in enumerate(ammp_is_variants):
            for j, lr in enumerate(lrs):
                m_q, m_avg, l_tot, l_one_iters, log_weights, entropies, ammp_is_times = ammp_is_results[(i,j)] #fn(T, post_params, init, lr, K)
                mean_errs, var_errs = get_errs(m_q, post_params)

                # print(f"Final mean of approx mean {fn.__name__}: ", fit_approx_post(m_q[-1])[:,0].abs().mean())
                # print(f"Final mean of approx var {fn.__name__}: ", fit_approx_post(m_q[-1])[:,1].mean())
                # print(f"Final mean error {fn.__name__}: ", mean_errs[-1])
                # print(f"Final var error {fn.__name__}: ", var_errs[-1])
                # print(f"Final ELBO {fn.__name__}: ", l_tot[-1])

                l0, = ax[0].plot(x_axis(ammp_is_times, use_time), mean_errs, c=colors[i], alpha = (j+1)/len(lrs), label=f'{fn.__name__}')

                l1, = ax[1].plot(x_axis(ammp_is_times, use_time), var_errs, c=colors[i], alpha = (j+1)/len(lrs), label=f'{fn.__name__}')

                l2, = ax[2].plot(x_axis(ammp_is_times[3:], use_time), [lt + lw for lt, lw in zip(l_one_iters, log_weights)][2:], c=colors[i], alpha = (j+1)/len(lrs), label=f'{fn.__name__}')

                l3, = ax[3].plot(x_axis(ammp_is_times, use_time), entropies, c=colors[i], alpha = (j+1)/len(lrs), label=f'{fn.__name__}')

                l4, = ax[4].plot(x_axis(ammp_is_times, use_time), [(fit_approx_post(m)[:,1] / post_params[:,1]).mean() for m in m_avg], c=colors[i], alpha = (j+1)/len(lrs), label=f'{fn.__name__}')

                if j == len(lrs) - 1:
                    final_lines.append(l0)

        hmc_mean_errs, hmc_var_errs = get_errs(hmc_moms, post_params)

        hmc_line, = ax[0].plot(x_axis(hmc_times, use_time)[:len(hmc_mean_errs)], hmc_mean_errs, c='black', label='HMC')
        final_lines.append(hmc_line)
        ax[1].plot(x_axis(hmc_times, use_time)[:len(hmc_mean_errs)], hmc_var_errs, c='black', label='HMC')

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


        ax[0].legend(loc='upper right')

        ax[-1].set_xlabel('Time (s)' if use_time else 'Iteration')
        if use_time:
            for a in ax:
                a.set_xlim(-0.001)

        ax[0].legend(handles = final_lines, loc='lower center', bbox_to_anchor=(0.5, 1.25), shadow=False, ncol=2)

        plt.tight_layout()  
        plt.savefig(f"figures/ammp_is_lrs/N{num_latents}_K{K}_{VAR_SIZE_STR}{'_TIME' if use_time else ''}.png")
        plt.close()

    print()