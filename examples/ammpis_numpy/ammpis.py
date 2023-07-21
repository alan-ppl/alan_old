import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import time 

import numpy as np
import math 
import torch as t

from torch.nn.functional import softplus
from torch.distributions import Normal
from torch.nn.functional import relu as ReLU

t.manual_seed(0)
t.cuda.manual_seed(0)

VERBOSE = False

def fit_approx_post(moments):
    # moments is a vector nx2 

    loc   = moments[:,0]
    raw_2nd_mom = moments[:,1] - loc**2
    bounded_2nd_mom = raw_2nd_mom + (-raw_2nd_mom + 1e-10)*(raw_2nd_mom<=0)
    scale = bounded_2nd_mom .sqrt()

    params = t.vstack([loc, scale]).t()

    return params


def IW(sample, params, post):
    # sample is a list of samples
    # params is a vector nx2
    # post is a torch distribution
    # returns a list of IW samples and the ELBO
    # print(sample.shape)
    logp = post.log_prob(sample)

    logq = Normal(params[:,0], params[:,1]).log_prob(sample)
    
    K = sample.shape[0]
    N = sample.shape[1]
    
    elbo = t.logsumexp((logp - logq), dim=0).sum() - N*math.log(K)
    
    lqp = logp - logq
    lqp_max = lqp.amax(axis=0)
    weights = t.exp(lqp - lqp_max)

    weights = weights / weights.sum(axis=0)
    
    Ex_one_iter = (weights * sample).sum(axis=0)
    Ex2_one_iter = (weights * sample**2).sum(axis=0)
    m_one_iter = t.stack([Ex_one_iter, Ex2_one_iter]).t()

    return m_one_iter, elbo


def sample(params, K=1):

    samps = Normal(params[:,0], params[:,1]).sample((K,))
    return samps

def entropy(params):

    return Normal(params[:,0], params[:,1]).entropy().sum()

 
def ammp_is(T, post, init, lr, K=5):
    m_q = [init]
    m_avg = [init]
    l_tot = [-1e15]
    log_w_t_minus_one = 0.0
    
    times = t.zeros(T+1)
    start_time = time.time()

    weights = [0]
    entropies = [entropy(fit_approx_post(m_q[-1]))]
    for i in range(T):
        Q_params = fit_approx_post(m_q[-1])

        z_t = sample(Q_params, K)

        m_one_iter_t, l_one_iter_t = IW(z_t, Q_params, post)

        H_Q = entropy(Q_params)
        H_Q_temp = entropy(Q_params)

        if VERBOSE and i % 100 == 0:
            print("Iteration: ", i, "ELBO: ", l_tot[-1])

        for _ in range(5):
            log_w_t = -(H_Q - H_Q_temp)
            #log_w_t = -ReLu(H_Q - H_Q_temp)

            weights.append(log_w_t)
            dt = log_w_t - log_w_t_minus_one

            l_tot_t = l_tot[-1] + dt + softplus(l_one_iter_t - dt - l_tot[-1])
            eta_t = np.exp(l_one_iter_t - l_tot_t)
          
            new_m_avg = eta_t * m_one_iter_t + (1 - eta_t) * m_avg[-1]

            Q_temp_params = fit_approx_post(new_m_avg)

            H_Q_temp = entropy(Q_temp_params)

        l_tot.append(l_tot_t)
        m_avg.append(new_m_avg)

        m_q.append(lr * new_m_avg + (1 - lr) * m_q[-1])

        times[i+1] = time.time() - start_time
        entropies.append(H_Q)
    return m_q, m_avg, l_tot, weights, entropies, times


def ammp_is_uniform_dt(T, post, init, lr, K=5):
    m_q = [init]
    m_avg = [init]
    l_tot = [-1e15]
    
    times = t.zeros(T+1)
    start_time = time.time()

    entropies = [entropy(fit_approx_post(m_q[-1]))]
    for i in range(T):
        Q_params = fit_approx_post(m_q[-1])

        z_t = sample(Q_params, K)

        m_one_iter_t, l_one_iter_t = IW(z_t, Q_params, post)

        if VERBOSE and i % 100 == 0:
            print("Iteration: ", i, "ELBO: ", l_tot[-1])

        dt = 0.0

        l_tot_t = l_tot[-1] + dt + softplus(l_one_iter_t - dt - l_tot[-1])
        eta_t = np.exp(l_one_iter_t - l_tot_t)

        new_m_avg = eta_t * m_one_iter_t + (1 - eta_t) * m_avg[-1]


        l_tot.append(l_tot_t)
        m_avg.append(new_m_avg)

        m_q.append(lr * new_m_avg + (1 - lr) * m_q[-1])

        times[i+1] = time.time() - start_time
        entropies.append(entropy(Q_params))

    return m_q, m_avg, l_tot, [0]*len(l_tot), entropies, times

def ammp_is_no_inner_loop(T, post, init, lr, K=5):
    m_q = [init]
    m_avg = [init]
    l_tot = [-1e15]
    weights = [0]
    times = t.zeros(T+1)
    start_time = time.time()

    H_Q_temp = entropy(fit_approx_post(m_q[-1]))
    entropies = [entropy(fit_approx_post(m_q[-1]))]
    for i in range(T):
        Q_params = fit_approx_post(m_q[-1]) 

        z_t = sample(Q_params, K)

        H_Q = entropy(Q_params)
        
        if VERBOSE and i % 100 == 0:
            print("Iteration: ", i, "ELBO: ", l_tot[-1])

        dt = -(H_Q - H_Q_temp) + 0.1
        #dt = -ReLu(H_Q - H_Q_temp)

        m_one_iter_t, l_one_iter_t = IW(z_t, Q_params, post)   

        l_tot_t = l_tot[-1] + dt + softplus(l_one_iter_t - dt - l_tot[-1])
        eta_t = np.exp(l_one_iter_t - l_tot_t)
        
        new_m_avg = eta_t * m_one_iter_t + (1 - eta_t) * m_avg[-1]

        l_tot.append(l_tot_t)
        m_avg.append(new_m_avg)

        m_q.append(lr * new_m_avg + (1 - lr) * m_q[-1])

        H_Q_temp = H_Q
        weights.append(dt)
        times[i+1] = time.time() - start_time
        entropies = H_Q

    return m_q, m_avg, l_tot, weights, entropies, times


def ammp_is_weight_all(T, post, init, lr, K=5):
    m_q = [init]
    m_one_iters = []
    l_one_iters = [-1e15]
    m_avg = [init]
    l_tot = [-1e15]

    times = t.zeros(T+1)
    start_time = time.time()

    entropies = [entropy(fit_approx_post(m_q[-1]))]
    for i in range(T):
        Q_params = fit_approx_post(m_q[-1]) 

        z_t = sample(Q_params, K)

        m_one_iter_t, l_one_iter_t = IW(z_t, Q_params, post)

        m_one_iters.append(m_one_iter_t)
        l_one_iters.append(l_one_iter_t)
        if VERBOSE and i % 100 == 0:
            print("Iteration: ", i, "ELBO: ", l_tot[-1])

        dts = [-ReLU(entropy(fit_approx_post(m)) - entropy(Q_params)) for m in m_q]
        lts = t.stack([lt - dt for lt, dt in zip(l_one_iters, dts)])

        l_tot.append(t.logsumexp(lts, dim=0))

        new_m_avg = sum([t.exp(lt - l_tot[-1]) * m for lt, m in zip(lts, m_one_iters)])

        m_avg.append(new_m_avg)

        m_q.append(lr * new_m_avg + (1 - lr) * m_q[-1])

        times[i+1] = time.time() - start_time

        entropies.append(entropy(Q_params))

    return m_q, m_avg, l_tot, [0] + dts, entropies, times

        
def mcmc(T, post, init, proposal_scale, burn_in=100):
    if type(proposal_scale) in (float, int):
        proposal_scale = proposal_scale*t.ones(num_latents, 1)

    N = init.shape[0]

    x = sample(init, 1)
    
    # to store samples
    samples = t.zeros((T + burn_in, N))
    samples[0,:] = x

    moments = [init]

    num_accepted = t.zeros(N)

    times = t.zeros(T+1)
    start_time = time.time()

    for i in range(T + burn_in):
        # normal proposal
        proposal_params = t.stack([x.squeeze(0), proposal_scale.squeeze(1)], dim=1)

        y = sample(proposal_params, 1)

        alpha = t.exp(post.log_prob(y) - post.log_prob(x))
        accepted = alpha > t.rand(N)
        num_accepted += accepted.squeeze(0)

        x = t.where(accepted, y, x)

        samples[i,:] = x

        # print(samples)
        # print((samples != 0).all(1).sum().item())
        # breakpoint()

        if i == burn_in-1:
            start_time = time.time()

        if i >= burn_in:
            Ex = t.mean(samples[burn_in:,:], dim=0)
            Ex2 = t.mean(samples[burn_in:,:]**2, dim=0)
            moments.append(t.stack([Ex, Ex2]).t())

            times[i-burn_in+1] = time.time() - start_time

    acceptance_rates = num_accepted / (T + burn_in)

    return moments, acceptance_rates.mean().item(), times

def lang(T, post, init, proposal_scale, burn_in=100):
    if type(proposal_scale) in (float, int):
        proposal_scale = proposal_scale*t.ones(num_latents, 1)

    N = init.shape[0]

    x = sample(init, 1)
    
    # to store samples
    samples = t.zeros((T + burn_in, N))
    samples[0,:] = x

    moments = [init]

    num_accepted = t.zeros(N)

    times = t.zeros(T+1)
    start_time = time.time()

    for i in range(T + burn_in):
        # get grad of log p(x)
        x.requires_grad_(True)
        x.retain_grad()
        logp_x = post.log_prob(x)
        logp_x.sum().backward()
        grad_logp_x = x.grad
        x.requires_grad_(False)

        # normal proposal
        # breakpoint()
        proposal_params = t.stack([(x + proposal_scale.t() * 0.5 * grad_logp_x).squeeze(), proposal_scale.squeeze(1)], dim=1)

        y = sample(proposal_params, 1)

        # get grad of log p(y)
        y.requires_grad_(True)
        y.retain_grad()
        logp_y = post.log_prob(y)
        logp_y.sum().backward()
        grad_logp_y = y.grad
        y.requires_grad_(False)

        reverse_proposal_params = t.stack([(y + proposal_scale.t() * 0.5 * grad_logp_y).squeeze(), proposal_scale.squeeze(1)], dim=1)


        alpha = t.exp(post.log_prob(y) - post.log_prob(x) + Normal(proposal_params[:,0], proposal_params[:,1]).log_prob(x) - Normal(reverse_proposal_params[:,0], reverse_proposal_params[:,1]).log_prob(y))
        accepted = alpha > t.rand(N)
        num_accepted += accepted.squeeze(0)

        x = t.where(accepted, y, x)

        samples[i,:] = x

        if i == burn_in-1:
            start_time = time.time()

        if i >= burn_in:
            Ex = t.mean(samples[burn_in:,:], dim=0)
            Ex2 = t.mean(samples[burn_in:,:]**2, dim=0)
            moments.append(t.stack([Ex, Ex2]).t())

            times[i-burn_in+1] = time.time() - start_time

    acceptance_rates = num_accepted / (T + burn_in)

    return moments, acceptance_rates.mean().item(), times

def get_errs(m_q, post):
    # m_q is a list of moments
    # post is a vector each row of which defines a posterior for a latent variable
    # returns a list of mean errors and a list of variance errors
    mean_errs = []
    var_errs = []

    for i in range(len(m_q)):
        mean_errs.append((post[:,0] - fit_approx_post(m_q[i])[:,0]).abs().mean())
        var_errs.append((post[:,1] - fit_approx_post(m_q[i])[:,1]).abs().mean())

    return mean_errs, var_errs


if __name__ == "__main__":
    num_latents = 500
    init = t.tensor([0.0,1.0], dtype=t.float64).repeat((num_latents,1))

    loc = Normal(0,150).sample((num_latents,1)).float()
    scale = Normal(0,0.1).sample((num_latents,1)).exp().float()

    post = t.cat([loc, scale], dim=1)
    post_dist = Normal(post[:,0], post[:,1])
    # breakpoint()

    colors = ['#a6611a','#dfc27d','#80cdc1','#018571']
    k = 0

    fig, ax = plt.subplots(3,1, figsize=(5.5, 8.0))


    for fn in [ammp_is, ammp_is_uniform_dt, ammp_is_no_inner_loop, ammp_is_weight_all]:
        m_q, m_avg, l_tot, log_weights, ammp_is_times = fn(1000, post_dist, init, 0.4, 100)
        mean_errs, var_errs = get_errs(m_q, post)

        print(f"Final ELBO {fn.__name__}: ", l_tot[-1])

        ax[0].plot(mean_errs, c=colors[k], label=f'{fn.__name__}')
        ax[0].set_title("Mean Error")

        ax[1].plot(var_errs, c=colors[k], label=f'{fn.__name__}')
        ax[1].set_title("Var Error")

        ax[2].plot([lt + lw for lt, lw in zip(l_tot, log_weights)][2:], c=colors[k], label=f'{fn.__name__}')
        ax[2].set_title("weighted log P_tot")
        k += 1


    
    m_mcmc, mcmc_acceptance_rate, mcmc_times = mcmc(1000, post_dist, init, 2.4*scale, burn_in=100)
    mean_errs_mcmc, var_errs_mcmc = get_errs(m_mcmc, post)
    print("MCMC acceptance rate: ", mcmc_acceptance_rate)  # should be 0.44 for Gaussian posterior

    ax[0].plot(mean_errs_mcmc, c='b', label='MCMC')

    ax[1].plot(var_errs_mcmc, c='b', label='MCMC')





    m_lang, lang_acceptance_rate, lang_times = lang(1000, post_dist, init, 2.4*scale, burn_in=100)
    mean_errs_lang, var_errs_lang = get_errs(m_lang, post)
    print("Lang acceptance rate: ", lang_acceptance_rate)  # should be 0.574 for Gaussian posterior

    ax[0].plot(mean_errs_lang, c='r', label='Lang')

    ax[1].plot(var_errs_lang, c='r', label='Lang')

    ax[0].legend(loc='upper right')

    plt.savefig('figures/ammp_is_vs_mcmc.png')
    # #Posterior mean and scale error
    # print("Posterior mean error: ", (post[:,0] - fit_approx_post(m_q[-1])[:,0]).abs().mean())
    # print("Posterior scale error: ", (post[:,1] - fit_approx_post(m_q[-1])[:,1]).abs().mean())

    # print("MCMC mean error: ", (post[:,0] - fit_approx_post(m_mcmc[-1])[:,0]).abs().mean())
    # print("MCMC scale error: ", (post[:,1] - fit_approx_post(m_mcmc[-1])[:,1]).abs().mean())

    # print("Lang mean error: ", (post[:,0] - fit_approx_post(m_lang[-1])[:,0]).abs().mean())
    # print("Lang scale error: ", (post[:,1] - fit_approx_post(m_lang[-1])[:,1]).abs().mean())

    # breakpoint()

    # print('Mean')
    # print(post[:,0])
    # print(fit_approx_post(m_q[-1])[:,0])
    # print(fit_approx_post(m_mcmc[-1])[:,0])


    # print('Scale')
    # print(post[:,1])
    # print(fit_approx_post(m_q[-1])[:,1])   
    # print(fit_approx_post(m_mcmc[-1])[:,1])
    



    # Want to tune MCMC to have acceptance rate of 0.44
    MCMC_SCALE_TEST = False
    if MCMC_SCALE_TEST:
        for proposal_scale in [2.4*scale, 2.4, 2.4/num_latents**0.5, 2.4*scale/num_latents**0.5, 2.4]:
            # N.B. 2.4*scale SHOULD be optimal (in Gaussian posterior case)
            m_mcmc, mcmc_acceptance_rate, mcmc_times = mcmc(10000, post, init, proposal_scale, burn_in=1000)

            print("Proposal sigma: ", proposal_scale)
            print("MCMC mean error: ", (post[:,0] - fit_approx_post(m_mcmc[-1])[:,0]).abs().mean())
            print("MCMC scale error: ", (post[:,1] - fit_approx_post(m_mcmc[-1])[:,1]).abs().mean())
            print("MCMC acceptance rate: ", mcmc_acceptance_rate)
            print()

    
        

