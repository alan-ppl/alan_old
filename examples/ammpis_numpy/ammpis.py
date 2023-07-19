import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import numpy as np
import math 
import torch as t

from torch.nn.functional import softplus
from torch.distributions import Normal


t.manual_seed(0)
t.cuda.manual_seed(0)

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
    # post is a vector nx2
    # returns a list of IW samples and the ELBO
    # print(sample.shape)
    logp = Normal(post[:,0], post[:,1]).log_prob(sample)

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
    for t in range(T):
        Q_params = fit_approx_post(m_q[-1])

        z_t = sample(Q_params, K)

        m_one_iter_t, l_one_iter_t = IW(z_t, Q_params, post)


        H_Q = entropy(Q_params)
        H_Q_temp = entropy(Q_params)

        if t % 100 == 0:
            print("Iteration: ", t, "ELBO: ", l_tot[-1])


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

    return m_q, m_avg, l_tot

        


if __name__ == "__main__":
    num_latents = 500
    init = t.tensor([0.0,1.0], dtype=t.float64).repeat((num_latents,1))

    loc = Normal(0,150).sample((num_latents,1)).float()
    scale = Normal(0,0.00001).sample((num_latents,1)).exp().float()
    post = t.cat([loc, scale], dim=1)
    m_q, m_avg, l_tot = ammp_is(1000,post, init, 0.4, 100)

    print("Final ELBO: ", l_tot[-1])

    #Posterior mean and scale error
    print("Posterior mean error: ", (post[:,0] - fit_approx_post(m_q[-1])[:,0]).abs().mean())
    print("Posterior scale error: ", (post[:,1] - fit_approx_post(m_q[-1])[:,1]).abs().mean())

    print('Mean')
    print(post[:,0])
    print(fit_approx_post(m_q[-1])[:,0])

    print('Scale')
    print(post[:,0])
    print(fit_approx_post(m_q[-1])[:,0])    

    
