from ammpis import *
import matplotlib.pyplot as plt
import torch as t
from torch.distributions import Normal, Uniform
import pickle
import sys
from get_posteriors import get_posteriors

T = 2000*(40)

posteriors = get_posteriors(T)

PLOT_ONLY = [False]*len(posteriors)

args = sys.argv[1:]

if len(args) > 0:
    PLOT_ONLY = [True]*len(posteriors)
    if args[0] != "-p":
        PLOT_ONLY[int(args[0])] = False

rws_lrs = [lambda i: 0.01,
           lambda i: (i+10)**(-0.75),
           lambda i: (i+10)**(-0.9)]

rws_names = ["rws, lr=0.01", "rws*, p=0.75", "rws*, p=0.9"]

ammp_is_variants = []#ammp_is]

lr = 0.4

results_collection = []
post_params_collection = []

for j, posterior_settings in enumerate(posteriors):
    if PLOT_ONLY[j]:
        pass

    else:
        results = {}
    
        num_latents  = posterior_settings["N"]
        K            = posterior_settings["K"]
        VAR_SIZE_STR = posterior_settings["VAR_SIZE"]
        LOC_VAR      = posterior_settings["LOC_VAR"]

        T_rws        = posterior_settings["T_rws"]
        T_vi         = posterior_settings["T_vi"]
        T_mcmc       = posterior_settings["T_mcmc"]
        T_lang       = posterior_settings["T_lang"]

        T_vi, T_mcmc, T_lang = T, T, T

        t.manual_seed(0)
        t.cuda.manual_seed(0)

        init = t.tensor([0.0,1.0], dtype=t.float64).repeat((num_latents,1))

        loc = Normal(0,LOC_VAR).sample((num_latents,1)).float()

        if VAR_SIZE_STR == "WIDE":
            scale = Normal(0,1).sample((num_latents,1)).exp().float()
        else:
            scale = Uniform(-6,-1).sample((num_latents,1)).exp().float()

        print(f'Location mean: {loc.abs().mean()}')
        print(f'Variance mean: {scale.mean()}')

        post_params = t.cat([loc, scale], dim=1)

        # first get the results
        # AMMPIS
        # ammp_is_variants = [ammp_is, ammp_is_ReLU, ammp_is_uniform_dt, ammp_is_no_inner_loop, ammp_is_no_inner_loop_ReLU]#, ammp_is_weight_all]
        for i, fn in enumerate(ammp_is_variants):
            m_q, m_avg, l_tot, l_one_iters, log_weights, entropies, ammp_is_times = fn(T, post_params, init, lr, K)
            mean_errs, var_errs = get_errs(m_q, post_params)
            results[fn.__name__] = [mean_errs, var_errs, ammp_is_times]
            print(f"{fn.__name__}, lr={lr} done.")

        # NATURAL RWS
        for i, lr_fn in enumerate(rws_lrs):
            m_q, l_one_iters, entropies, rws_times = natural_rws(T_rws, post_params, init, lr_fn, K)
            mean_errs, var_errs = get_errs(m_q, post_params)
            results[f"rws{i}"] = [mean_errs, var_errs, rws_times]
            print(f"rws{i} done.")

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

        hmc_mean_errs, hmc_var_errs = get_errs(hmc_moms, post_params)
        results["HMC"] = [hmc_mean_errs, hmc_var_errs, hmc_times]

        # VI
        vi_means, vi_vars, elbos, entropies, vi_times = VI(T_vi, post_params, init, 0.05, K=K)
        mean_errs = []
        var_errs = []

        for i in range(len(vi_means)):
            mean_errs.append((post_params[:,0] - vi_means[i]).abs().mean())
            var_errs.append((post_params[:,1] - vi_vars[i].exp()).abs().mean())

        results["VI"] = [mean_errs, var_errs, vi_times]
        print("VI done.")

        # MCMC
        m_mcmc, mcmc_acceptance_rate, mcmc_times, mcmc_samples = mcmc(T_mcmc, post_params, init, 2.4*scale, burn_in=T//10)
        mean_errs_mcmc, var_errs_mcmc = get_errs(m_mcmc, post_params)
        results["mcmc"] = [mean_errs_mcmc, var_errs_mcmc, mcmc_times]
        print(f"MCMC done. Acceptance = {mcmc_acceptance_rate:.4f} (ideal ~= 0.44)")

        # LANG
        m_lang, lang_acceptance_rate, lang_times, lang_samples = lang(T_lang, post_params, 1e-14*init, 0.5*scale**0.0, burn_in=T//10)
        mean_errs_lang, var_errs_lang = get_errs(m_lang, post_params)
        results["lang"] = [mean_errs_lang, var_errs_lang, lang_times]
        print(f"Lang done. Acceptance = {lang_acceptance_rate:.4f} (ideal ~= 0.574)")

        print(results["lang"][0][100], results["lang"][1][100])

        # breakpoint()

        with open(f"saved_results/increasing_difficulty{j}_EXTRA.pkl", "wb") as f:
            pickle.dump({"results": results, "post_params": post_params}, f)

        print(f"Posterior {j} results saved.")
