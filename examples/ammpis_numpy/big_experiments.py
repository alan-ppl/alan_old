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
lr = 0.4

Ns = [50, 500, 5000]
Ks = [3, 10, 30]#, 100]
var_sizes = ["WIDE", "NARROW"]

total_num_plots = len(Ns) * len(Ks) * len(var_sizes)

plot_counter = 0
for VAR_SIZE_STR in var_sizes:

    results = {N: {K: {} for K in Ks} for N in Ns}
    post_params_collection = {N: {K: None for K in Ks} for N in Ns}
    for num_latents, K in product(Ns, Ks):
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
        post_params_collection[num_latents][K] = post_params

        # first get the results
        # AMMPIS
        ammp_is_variants = [ammp_is, ammp_is_ReLU, ammp_is_uniform_dt, ammp_is_no_inner_loop, ammp_is_no_inner_loop_ReLU]#, ammp_is_weight_all]

        for i, fn in enumerate(ammp_is_variants):
            results[num_latents][K][fn.__name__] = fn(T, post_params, init, lr, K)
            print(f"{fn.__name__}, lr={lr} done.")

        # NATURAL RWS
        rws_variants = [natural_rws, natural_rws_difference, natural_rws_standardised]
        rws_lrs = [0.01, 0.01, 0.4]
        for i, fn in enumerate(rws_variants):
            results[num_latents][K][fn.__name__] = fn(T, post_params, init, rws_lrs[i], K)
            print(f"{fn.__name__}, lr={rws_lrs[i]} done.")

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
            
            print("HMC done.")

        results[num_latents][K]["HMC"] = [hmc_moms, hmc_times, hmc_samples]

        # VI
        results[num_latents][K]["VI"] = VI(T, post_params, init, 0.05, K=K)
        print("VI done.")

        # MCMC
        results[num_latents][K]["mcmc"] = mcmc(T, post_params, init, 2.4*scale, burn_in=T//10)
        print("MCMC done.")

        # LANG
        results[num_latents][K]["lang"] = lang(T, post_params, init, 1e-6*scale, burn_in=T//10)
        print("Lang done.")


        # Now to actually plot the results in a couple of ways
        # First: one plot per num_latents-K-posterior_width combination

        for use_time in [True, False]:
            fig, ax = plt.subplots(5,1, figsize=(5.5, 14.0))

            # AMMPIS
            colours = ['#543005','#8c510a','#bf812d','#dfc27d', "pink", "#FF69B4", "#f6e8c3", "#c7eae5", "#80cdc1"]

            for i, fn in enumerate(ammp_is_variants):
                m_q, m_avg, l_tot, l_one_iters, log_weights, entropies, ammp_is_times = results[num_latents][K][fn.__name__] 
                mean_errs, var_errs = get_errs(m_q, post_params)

                ax[0].plot(x_axis(ammp_is_times, use_time), mean_errs, c=colours[i], label=f'{fn.__name__}')
                ax[1].plot(x_axis(ammp_is_times, use_time), var_errs, c=colours[i], label=f'{fn.__name__}')
                ax[2].plot(x_axis(ammp_is_times[3:], use_time), [lt + lw for lt, lw in zip(l_one_iters, log_weights)][2:], c=colours[i], label=f'{fn.__name__}')
                ax[3].plot(x_axis(ammp_is_times, use_time), entropies, c=colours[i], label=f'{fn.__name__}')
                ax[4].plot(x_axis(ammp_is_times, use_time), [(fit_approx_post(m)[:,1] / post_params[:,1]).mean() for m in m_avg], c=colours[i], label=f'{fn.__name__}')


            # NATURAL RWS
            rws_colours = ['#80cdc1','#35978f','#01665e']

            for i, fn in enumerate(rws_variants):
                m_q, l_one_iters, entropies, times = results[num_latents][K][fn.__name__] 
                mean_errs, var_errs = get_errs(m_q, post_params)

                ax[0].plot(x_axis(times, use_time), mean_errs, c=rws_colours[i], label=f'{fn.__name__}')
                ax[1].plot(x_axis(times, use_time), var_errs, c=rws_colours[i], label=f'{fn.__name__}')
                ax[2].plot(x_axis(times[3:], use_time), [lt for lt in l_one_iters][2:], c=rws_colours[i], label=f'{fn.__name__}')
                ax[3].plot(x_axis(times, use_time), entropies, c=rws_colours[i], label=f'{fn.__name__}')
                ax[4].plot(x_axis(times, use_time), [(fit_approx_post(m)[:,1] / post_params[:,1]).mean() for m in m_q], c=rws_colours[i], label=f'{fn.__name__}')


            # HMC
            hmc_moms, hmc_times, hmc_samples = results[num_latents][K]["HMC"]
            hmc_mean_errs, hmc_var_errs = get_errs(hmc_moms, post_params)

            ax[0].plot(x_axis(hmc_times, use_time)[:len(hmc_mean_errs)], hmc_mean_errs, c='black', label='HMC')
            ax[1].plot(x_axis(hmc_times, use_time)[:len(hmc_var_errs)], hmc_var_errs, c='black', label='HMC')

            if not use_time:
                # VI
                vi_means, vi_vars, elbos, entropies, vi_times = results[num_latents][K]["VI"]

                mean_errs = []
                var_errs = []

                for i in range(len(vi_means)):
                    mean_errs.append((post_params[:,0] - vi_means[i]).abs().mean())
                    var_errs.append((post_params[:,1] - vi_vars[i].exp()).abs().mean())

                ax[0].plot(x_axis(vi_times[:-1]), mean_errs, c='#54278f', label='VI')
                ax[1].plot(x_axis(vi_times[:-1]), var_errs, c='#54278f', label='VI')
                ax[2].plot(x_axis(vi_times[:-1]), elbos, c='#54278f', label='VI') 
                ax[3].plot(x_axis(vi_times[:-1]), entropies, c='#54278f', label='VI')
                ax[4].plot(x_axis(vi_times[:-1]), [(v.exp() / post_params[:,1]).mean() for v in vi_vars], c='#54278f', label='VI')

                # MCMC
                m_mcmc, mcmc_acceptance_rate, mcmc_times, mcmc_samples = results[num_latents][K]["mcmc"]
                mean_errs_mcmc, var_errs_mcmc = get_errs(m_mcmc, post_params)

                ax[0].plot(x_axis(mcmc_times), mean_errs_mcmc, c='b', label='MCMC')
                ax[1].plot(x_axis(mcmc_times), var_errs_mcmc, c='b', label='MCMC')

                # LANG
                m_lang, lang_acceptance_rate, lang_times, lang_samples = results[num_latents][K]["lang"]
                mean_errs_lang, var_errs_lang = get_errs(m_lang, post_params)

                ax[0].plot(x_axis(lang_times), mean_errs_lang, c='r', label='Lang')
                ax[1].plot(x_axis(lang_times), var_errs_lang, c='r', label='Lang')


            # Formatting

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


            # ax[0].legend(loc='upper right')

            ax[-1].set_xlabel('Time (s)' if use_time else 'Iteration')
            if use_time:
                for a in ax:
                    a.set_xlim(-0.001)

            ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.25), shadow=False, ncol=2)

            plt.tight_layout()  
            plt.savefig(f"figures/big_experiments/N{num_latents}_K{K}_{VAR_SIZE_STR}{'_TIME' if use_time else ''}.png")
            plt.close()

    print()

    # Now for the other plots:
    #   1. The regular 5 rows but repeated over columns with increasing N for fixed K
    #           (i.e. so the posterior gets more difficult from left to right)
    #   2. Again the regular 5 rows but repeated over increasing K for fixed N
    #           (so the ammpis and rws methods should improve from left to right)

    for K in Ks:
        for use_time in [True, False]:
            fig, ax = plt.subplots(5, len(Ns), figsize=(16.5, 14.0))

            for j, num_latents in enumerate(Ns):
                post_params = post_params_collection[num_latents][K]

                # AMMPIS
                colours = ['#543005','#8c510a','#bf812d','#dfc27d', "pink", "#FF69B4", "#f6e8c3", "#c7eae5", "#80cdc1"]

                for i, fn in enumerate(ammp_is_variants):
                    m_q, m_avg, l_tot, l_one_iters, log_weights, entropies, ammp_is_times = results[num_latents][K][fn.__name__] 
                    mean_errs, var_errs = get_errs(m_q, post_params)

                    ax[0, j].plot(x_axis(ammp_is_times, use_time), mean_errs, c=colours[i], label=f'{fn.__name__}')
                    ax[1, j].plot(x_axis(ammp_is_times, use_time), var_errs, c=colours[i], label=f'{fn.__name__}')
                    ax[2, j].plot(x_axis(ammp_is_times[3:], use_time), [lt + lw for lt, lw in zip(l_one_iters, log_weights)][2:], c=colours[i], label=f'{fn.__name__}')
                    ax[3, j].plot(x_axis(ammp_is_times, use_time), entropies, c=colours[i], label=f'{fn.__name__}')
                    ax[4, j].plot(x_axis(ammp_is_times, use_time), [(fit_approx_post(m)[:,1] / post_params[:,1]).mean() for m in m_avg], c=colours[i], label=f'{fn.__name__}')


                # NATURAL RWS
                rws_colours = ['#80cdc1','#35978f','#01665e']

                for i, fn in enumerate(rws_variants):
                    m_q, l_one_iters, entropies, times = results[num_latents][K][fn.__name__] 
                    mean_errs, var_errs = get_errs(m_q, post_params)

                    ax[0, j].plot(x_axis(times, use_time), mean_errs, c=rws_colours[i], label=f'{fn.__name__}')
                    ax[1, j].plot(x_axis(times, use_time), var_errs, c=rws_colours[i], label=f'{fn.__name__}')
                    ax[2, j].plot(x_axis(times[3:], use_time), [lt for lt in l_one_iters][2:], c=rws_colours[i], label=f'{fn.__name__}')
                    ax[3, j].plot(x_axis(times, use_time), entropies, c=rws_colours[i], label=f'{fn.__name__}')
                    ax[4, j].plot(x_axis(times, use_time), [(fit_approx_post(m)[:,1] / post_params[:,1]).mean() for m in m_q], c=rws_colours[i], label=f'{fn.__name__}')


                # HMC
                hmc_moms, hmc_times, hmc_samples = results[num_latents][K]["HMC"]
                hmc_mean_errs, hmc_var_errs = get_errs(hmc_moms, post_params)

                ax[0, j].plot(x_axis(hmc_times, use_time)[:len(hmc_mean_errs)], hmc_mean_errs, c='black', label='HMC')
                ax[1, j].plot(x_axis(hmc_times, use_time)[:len(hmc_var_errs)], hmc_var_errs, c='black', label='HMC')

                if not use_time:
                    # VI
                    vi_means, vi_vars, elbos, entropies, vi_times = results[num_latents][K]["VI"]

                    mean_errs = []
                    var_errs = []

                    for i in range(len(vi_means)):
                        mean_errs.append((post_params[:,0] - vi_means[i]).abs().mean())
                        var_errs.append((post_params[:,1] - vi_vars[i].exp()).abs().mean())

                    ax[0, j].plot(x_axis(vi_times[:-1]), mean_errs, c='#54278f', label='VI')
                    ax[1, j].plot(x_axis(vi_times[:-1]), var_errs, c='#54278f', label='VI')
                    ax[2, j].plot(x_axis(vi_times[:-1]), elbos, c='#54278f', label='VI') 
                    ax[3, j].plot(x_axis(vi_times[:-1]), entropies, c='#54278f', label='VI')
                    ax[4, j].plot(x_axis(vi_times[:-1]), [(v.exp() / post_params[:,1]).mean() for v in vi_vars], c='#54278f', label='VI')


                    # MCMC
                    m_mcmc, mcmc_acceptance_rate, mcmc_times, mcmc_samples = results[num_latents][K]["mcmc"]
                    mean_errs_mcmc, var_errs_mcmc = get_errs(m_mcmc, post_params)

                    ax[0, j].plot(x_axis(mcmc_times), mean_errs_mcmc, c='b', label='MCMC')
                    ax[1, j].plot(x_axis(mcmc_times), var_errs_mcmc, c='b', label='MCMC')

                    # LANG
                    m_lang, lang_acceptance_rate, lang_times, lang_samples = results[num_latents][K]["lang"]
                    mean_errs_lang, var_errs_lang = get_errs(m_lang, post_params)

                    ax[0, j].plot(x_axis(lang_times), mean_errs_lang, c='r', label='Lang')
                    ax[1, j].plot(x_axis(lang_times), var_errs_lang, c='r', label='Lang')


                # Formatting
                ax[0, j].set_title(f"Mean Error, N={num_latents}")
                ax[1, j].set_title("Var Error")
                ax[2, j].set_title("weighted l_one_iter")
                ax[3, j].set_title("Entropy")
                ax[4, j].set_title("Q variance / posterior variance")

                # log y axis
                ax[0, j].set_yscale('log')
                ax[1, j].set_yscale('log')
                # ax[2].set_yscale('symlog')
                # ax[3].set_yscale('symlog')

                ax[2, j].set_ylim(-50, 20)
                ax[4, j].set_ylim(0.5, 1.5)


                # ax[0].legend(loc='upper right')

                ax[-1, j].set_xlabel('Time (s)' if use_time else 'Iteration')
                if use_time:
                    for a in ax[:,j]:
                        a.set_xlim(-0.001)

            ax[0, 0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.25), shadow=False, ncol=2)

            plt.tight_layout()  
            plt.savefig(f"figures/big_experiments/VARIED_N/K{K}_{VAR_SIZE_STR}{'_TIME' if use_time else ''}.png")
            plt.savefig(f"figures/big_experiments/VARIED_N/K{K}_{VAR_SIZE_STR}{'_TIME' if use_time else ''}.pdf")
            plt.close()

    print("Varying N plot saved.")

    for num_latents in Ns:
        for use_time in [True, False]:
            fig, ax = plt.subplots(5, len(Ks), figsize=(16.5, 14.0))

            for j, K in enumerate(Ks):
                post_params = post_params_collection[num_latents][K]

                # AMMPIS
                colours = ['#543005','#8c510a','#bf812d','#dfc27d', "pink", "#FF69B4", "#f6e8c3", "#c7eae5", "#80cdc1"]

                for i, fn in enumerate(ammp_is_variants):
                    m_q, m_avg, l_tot, l_one_iters, log_weights, entropies, ammp_is_times = results[num_latents][K][fn.__name__] 
                    mean_errs, var_errs = get_errs(m_q, post_params)

                    ax[0, j].plot(x_axis(ammp_is_times, use_time), mean_errs, c=colours[i], label=f'{fn.__name__}')
                    ax[1, j].plot(x_axis(ammp_is_times, use_time), var_errs, c=colours[i], label=f'{fn.__name__}')
                    ax[2, j].plot(x_axis(ammp_is_times[3:], use_time), [lt + lw for lt, lw in zip(l_one_iters, log_weights)][2:], c=colours[i], label=f'{fn.__name__}')
                    ax[3, j].plot(x_axis(ammp_is_times, use_time), entropies, c=colours[i], label=f'{fn.__name__}')
                    ax[4, j].plot(x_axis(ammp_is_times, use_time), [(fit_approx_post(m)[:,1] / post_params[:,1]).mean() for m in m_avg], c=colours[i], label=f'{fn.__name__}')


                # NATURAL RWS
                rws_colours = ['#80cdc1','#35978f','#01665e']

                for i, fn in enumerate(rws_variants):
                    m_q, l_one_iters, entropies, times = results[num_latents][K][fn.__name__] 
                    mean_errs, var_errs = get_errs(m_q, post_params)

                    ax[0, j].plot(x_axis(times, use_time), mean_errs, c=rws_colours[i], label=f'{fn.__name__}')
                    ax[1, j].plot(x_axis(times, use_time), var_errs, c=rws_colours[i], label=f'{fn.__name__}')
                    ax[2, j].plot(x_axis(times[3:], use_time), [lt for lt in l_one_iters][2:], c=rws_colours[i], label=f'{fn.__name__}')
                    ax[3, j].plot(x_axis(times, use_time), entropies, c=rws_colours[i], label=f'{fn.__name__}')
                    ax[4, j].plot(x_axis(times, use_time), [(fit_approx_post(m)[:,1] / post_params[:,1]).mean() for m in m_q], c=rws_colours[i], label=f'{fn.__name__}')


                # HMC
                hmc_moms, hmc_times, hmc_samples = results[num_latents][K]["HMC"]
                hmc_mean_errs, hmc_var_errs = get_errs(hmc_moms, post_params)

                ax[0, j].plot(x_axis(hmc_times, use_time)[:len(hmc_mean_errs)], hmc_mean_errs, c='black', label='HMC')
                ax[1, j].plot(x_axis(hmc_times, use_time)[:len(hmc_var_errs)], hmc_var_errs, c='black', label='HMC')

                if not use_time:
                    # VI
                    vi_means, vi_vars, elbos, entropies, vi_times = results[num_latents][K]["VI"]

                    mean_errs = []
                    var_errs = []

                    for i in range(len(vi_means)):
                        mean_errs.append((post_params[:,0] - vi_means[i]).abs().mean())
                        var_errs.append((post_params[:,1] - vi_vars[i].exp()).abs().mean())

                    ax[0, j].plot(x_axis(vi_times[:-1]), mean_errs, c='#54278f', label='VI')
                    ax[1, j].plot(x_axis(vi_times[:-1]), var_errs, c='#54278f', label='VI')
                    ax[2, j].plot(x_axis(vi_times[:-1]), elbos, c='#54278f', label='VI') 
                    ax[3, j].plot(x_axis(vi_times[:-1]), entropies, c='#54278f', label='VI')
                    ax[4, j].plot(x_axis(vi_times[:-1]), [(v.exp() / post_params[:,1]).mean() for v in vi_vars], c='#54278f', label='VI')


                    # MCMC
                    m_mcmc, mcmc_acceptance_rate, mcmc_times, mcmc_samples = results[num_latents][K]["mcmc"]
                    mean_errs_mcmc, var_errs_mcmc = get_errs(m_mcmc, post_params)

                    ax[0, j].plot(x_axis(mcmc_times), mean_errs_mcmc, c='b', label='MCMC')
                    ax[1, j].plot(x_axis(mcmc_times), var_errs_mcmc, c='b', label='MCMC')

                    # LANG
                    m_lang, lang_acceptance_rate, lang_times, lang_samples = results[num_latents][K]["lang"]
                    mean_errs_lang, var_errs_lang = get_errs(m_lang, post_params)

                    ax[0, j].plot(x_axis(lang_times), mean_errs_lang, c='r', label='Lang')
                    ax[1, j].plot(x_axis(lang_times), var_errs_lang, c='r', label='Lang')


                # Formatting
                ax[0, j].set_title(f"Mean Error, K={K}")
                ax[1, j].set_title("Var Error")
                ax[2, j].set_title("weighted l_one_iter")
                ax[3, j].set_title("Entropy")
                ax[4, j].set_title("Q variance / posterior variance")

                # log y axis
                ax[0, j].set_yscale('log')
                ax[1, j].set_yscale('log')
                # ax[2].set_yscale('symlog')
                # ax[3].set_yscale('symlog')

                ax[2, j].set_ylim(-50, 20)
                ax[4, j].set_ylim(0.5, 1.5)


                # ax[0].legend(loc='upper right')

                ax[-1, j].set_xlabel('Time (s)' if use_time else 'Iteration')
                if use_time:
                    for a in ax[:,j]:
                        a.set_xlim(-0.001)

            ax[0, 0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.25), shadow=False, ncol=2)

            plt.tight_layout()  
            plt.savefig(f"figures/big_experiments/VARIED_K/N{num_latents}_{VAR_SIZE_STR}{'_TIME' if use_time else ''}.png")
            plt.savefig(f"figures/big_experiments/VARIED_K/N{num_latents}_{VAR_SIZE_STR}{'_TIME' if use_time else ''}.pdf")
            plt.close()

    print("Varying K plot saved.")

    print()