from ammpis import *
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import torch as t

from torch.distributions import Normal, Laplace, Gumbel

t.manual_seed(1)
t.cuda.manual_seed(0)

X_AXIS_TIME = False

def x_axis(xs):
    if X_AXIS_TIME:
        return xs
    else:
        return range(len(xs))

T = 2000
num_latents = 500
K=100
init = t.tensor([0.0,1.0], dtype=t.float64).repeat((num_latents,1))

loc = Normal(0,1).sample((num_latents,1)).float()
# scale = Normal(0,1).sample((num_latents,1)).exp().float()
scale = Uniform(-5,-0.5).sample((num_latents,1)).exp().float()

# print(scale)
# breakpoint()
print(f'Location mean: {loc.abs().mean()}')
print(f'Variance mean: {scale.mean()}')
post_params = t.cat([loc, scale], dim=1)
# breakpoint()

colors = ['#543005','#8c510a','#bf812d','#dfc27d', "pink", "#FF69B4", "#f6e8c3", "#c7eae5", "#80cdc1"]
k = 0

fig, ax = plt.subplots(5,1, figsize=(7, 16.0))


for fn in [ammp_is, ammp_is_ReLU, ammp_is_uniform_dt, ammp_is_no_inner_loop, ammp_is_no_inner_loop_ReLU]:#, ammp_is_weight_all]:
    m_q, m_avg, l_tot, l_one_iters, log_weights, entropies, ammp_is_times = fn(T, post_params, init, 0.05, K)
    mean_errs, var_errs = get_errs(m_q, post_params)

    print(f"Final mean of approx mean {fn.__name__}: ", fit_approx_post(m_q[-1])[:,0].abs().mean())
    print(f"Final mean of approx var {fn.__name__}: ", fit_approx_post(m_q[-1])[:,1].mean())
    print(f"Final mean error {fn.__name__}: ", mean_errs[-1])
    print(f"Final var error {fn.__name__}: ", var_errs[-1])
    print(f"Final ELBO {fn.__name__}: ", l_tot[-1])

    ax[0].plot(x_axis(ammp_is_times), mean_errs, c=colors[k], label=f'{fn.__name__}')
    ax[0].set_title("Mean Error")

    ax[1].plot(x_axis(ammp_is_times), var_errs, c=colors[k], label=f'{fn.__name__}')
    ax[1].set_title("Var Error")

    ax[2].plot(x_axis(ammp_is_times[3:]), [lt + lw for lt, lw in zip(l_one_iters, log_weights)][2:], c=colors[k], label=f'{fn.__name__}')
    ax[2].set_title("weighted l_one_iter")

    ax[3].plot(x_axis(ammp_is_times), entropies, c=colors[k], label=f'{fn.__name__}')
    ax[3].set_title("Entropy")

    ax[4].plot(x_axis(ammp_is_times), [(fit_approx_post(m)[:,1] / post_params[:,1]).mean() for m in m_avg], c=colors[k], label=f'{fn.__name__}')
    ax[4].set_title("Q variance / posterior variance")
    k += 1

rws_colours = ['#80cdc1','#35978f','#01665e']
rws_lrs = [0.4, 0.01, 0.1]

for i, fn in enumerate([natural_rws, natural_rws_difference, natural_rws_standardised]):
    m_q, l_one_iters, entropies, times = fn(T, post_params, init, rws_lrs[i], K)
    mean_errs, var_errs = get_errs(m_q, post_params)

    ax[0].plot(x_axis(times), mean_errs, c=rws_colours[i], label=f'{fn.__name__}')
    ax[1].plot(x_axis(times), var_errs, c=rws_colours[i], label=f'{fn.__name__}')
    ax[2].plot(x_axis(times[3:]), [lt for lt in l_one_iters][2:], c=rws_colours[i], label=f'{fn.__name__}')
    ax[3].plot(x_axis(times), entropies, c=rws_colours[i], label=f'{fn.__name__}')
    ax[4].plot(x_axis(times), [(fit_approx_post(m)[:,1] / post_params[:,1]).mean() for m in m_q], c=rws_colours[i], label=f'{fn.__name__}')


vi_means, vi_vars, elbos, entropies, vi_times = VI(T, post_params, init, 0.05, K=K)

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

hmc_params   = {"N": num_latents,
                "T": T//4,
                "post_params": post_params,
                "init": init,
                "post_type": Normal,
                "num_chains": 4}

with open('saved_hmc.pkl', 'rb') as f:
    saved_hmc = pickle.load(f)

# params_match = {key: hmc_params[key] == saved_hmc["params"][key] for key in hmc_params}
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

mean_errs, var_errs = get_errs(hmc_moms, post_params)
ax[0].plot(x_axis(hmc_times), mean_errs, c='black', label='HMC')
ax[1].plot(x_axis(hmc_times), var_errs, c='black', label='HMC')


m_mcmc, mcmc_acceptance_rate, mcmc_times, mcmc_samples = mcmc(T, post_params, init, 2.4*scale, burn_in=T//10)
mean_errs_mcmc, var_errs_mcmc = get_errs(m_mcmc, post_params)
print("MCMC acceptance rate: ", mcmc_acceptance_rate)  # should be 0.44 for Gaussian posterior


m_lang, lang_acceptance_rate, lang_times, lang_samples = lang(T, post_params, init, 1e-6*scale, burn_in=T//10)
mean_errs_lang, var_errs_lang = get_errs(m_lang, post_params)
print("Lang acceptance rate: ", lang_acceptance_rate)  # should be 0.574 for Gaussian posterior

ax[0].plot(x_axis(mcmc_times), mean_errs_mcmc, c='b', label='MCMC')
ax[1].plot(x_axis(mcmc_times), var_errs_mcmc, c='b', label='MCMC')

ax[0].plot(x_axis(lang_times), mean_errs_lang, c='r', label='Lang')
ax[1].plot(x_axis(lang_times), var_errs_lang, c='r', label='Lang')

# log y axis
ax[0].set_yscale('log')
ax[1].set_yscale('log')
# ax[2].set_yscale('symlog')
# ax[3].set_yscale('symlog')

ax[2].set_ylim(-50, 20)
ax[4].set_ylim(0, 10)


# ax[0].legend(loc='upper right')
ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.25), shadow=False, ncol=3)

ax[-1].set_xlabel('Time (s)' if X_AXIS_TIME else 'Iteration')
if X_AXIS_TIME:
    for a in ax:
        a.set_xlim(-0.001)

plt.tight_layout()  
plt.savefig('figures/ammp_is_vs_mcmc.png')


# #Posterior mean and scale error
# print("Posterior mean error: ", (post_params[:,0] - fit_approx_post(m_q[-1])[:,0]).abs().mean())
# print("Posterior scale error: ", (post_params[:,1] - fit_approx_post(m_q[-1])[:,1]).abs().mean())

# print("MCMC mean error: ", (post_params[:,0] - fit_approx_post(m_mcmc[-1])[:,0]).abs().mean())
# print("MCMC scale error: ", (post_params[:,1] - fit_approx_post(m_mcmc[-1])[:,1]).abs().mean())

# print("Lang mean error: ", (post_params[:,0] - fit_approx_post(m_lang[-1])[:,0]).abs().mean())
# print("Lang scale error: ", (post_params[:,1] - fit_approx_post(m_lang[-1])[:,1]).abs().mean())

# breakpoint()

# print('Mean')
# print(post_params[:,0])
# print(fit_approx_post(m_q[-1])[:,0])
# print(fit_approx_post(m_mcmc[-1])[:,0])


# print('Scale')
# print(post_params[:,1])
# print(fit_approx_post(m_q[-1])[:,1])   
# print(fit_approx_post(m_mcmc[-1])[:,1])

# Want to tune MCMC to have acceptance rate of 0.44
MCMC_SCALE_TEST = False
if MCMC_SCALE_TEST:
    for proposal_scale in [2.4*scale, 2.4, 2.4/num_latents**0.5, 2.4*scale/num_latents**0.5, 2.4]:
        # N.B. 2.4*scale SHOULD be optimal (in Gaussian posterior case)
        m_mcmc, mcmc_acceptance_rate, mcmc_times, mcmc_samples = mcmc(10000, post_params, init, proposal_scale, burn_in=1000)

        print("Proposal sigma: ", proposal_scale)
        print("MCMC mean error: ", (post_params[:,0] - fit_approx_post(m_mcmc[-1])[:,0]).abs().mean())
        print("MCMC scale error: ", (post_params[:,1] - fit_approx_post(m_mcmc[-1])[:,1]).abs().mean())
        print("MCMC acceptance rate: ", mcmc_acceptance_rate)
        print()