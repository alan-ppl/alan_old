import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import time 

import numpy as np
import math 
import torch as t

from torch.nn.functional import softplus
from torch.distributions import Normal

t.manual_seed(0)
t.cuda.manual_seed(0)

VERBOSE = False

def fit_approx_post(moments):
    # moments is a vector nx2 

    loc   = moments[:,0]
    a = moments[:,1] - loc**2
    A = a + (-a + 1e-10)*(a<=0)
    scale = A.sqrt()
    
    params = t.cat([loc.unsqueeze(1), scale.unsqueeze(1)], dim=1)

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
    m_one_iter = t.cat([Ex_one_iter.unsqueeze(1), Ex2_one_iter.unsqueeze(1)], dim=1)
    
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

    for i in range(T):
        Q_params = fit_approx_post(m_q[-1])

        z_t = sample(Q_params, K)

        m_one_iter_t, l_one_iter_t = IW(z_t, Q_params, post)


        H_Q = entropy(Q_params)
        H_Q_temp = entropy(Q_params)

        if VERBOSE and i % 100 == 0:
            print("Iteration: ", i, "ELBO: ", l_tot[-1])


        for _ in range(1):
            log_w_t = -(H_Q - H_Q_temp)
            #log_w_t = -ReLu(H_Q - H_Q_temp)

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

    return m_q, m_avg, l_tot, times

        
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
            moments.append(t.cat([Ex.unsqueeze(1), Ex2.unsqueeze(1)], dim=1))

            times[i-burn_in+1] = time.time() - start_time


    acceptance_rates = num_accepted / (T + burn_in)

    return moments, acceptance_rates.mean().item(), times



if __name__ == "__main__":
    num_latents = 500
    init = t.tensor([0.0,1.0], dtype=t.float64).repeat((num_latents,1))

    loc = Normal(0,150).sample((num_latents,1)).float()
    scale = Normal(0,0.1).sample((num_latents,1)).exp().float()

    post = t.cat([loc, scale], dim=1)
    post_dist = Normal(post[:,0], post[:,1])
    # breakpoint()

    m_q, m_avg, l_tot, ammp_is_times = ammp_is(1000, post_dist, init, 0.4, 100)

    print("Final ELBO: ", l_tot[-1])

    m_mcmc, mcmc_acceptance_rate, mcmc_times = mcmc(10000, post_dist, init, 2.4*scale, burn_in=1000)
    print("MCMC acceptance rate: ", mcmc_acceptance_rate)

    #Posterior mean and scale error
    print("Posterior mean error: ", (post[:,0] - fit_approx_post(m_q[-1])[:,0]).abs().mean())
    print("Posterior scale error: ", (post[:,1] - fit_approx_post(m_q[-1])[:,1]).abs().mean())

    print("MCMC mean error: ", (post[:,0] - fit_approx_post(m_mcmc[-1])[:,0]).abs().mean())
    print("MCMC scale error: ", (post[:,1] - fit_approx_post(m_mcmc[-1])[:,1]).abs().mean())


    print('Mean')
    print(post[:,0])
    print(fit_approx_post(m_q[-1])[:,0])
    print(fit_approx_post(m_mcmc[-1])[:,0])


    print('Scale')
    print(post[:,1])
    print(fit_approx_post(m_q[-1])[:,1])   
    print(fit_approx_post(m_mcmc[-1])[:,1])
    

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

    
        

