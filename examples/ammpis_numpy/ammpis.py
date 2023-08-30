import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import time 

import stan
import numpy as np
import math 
import torch as t
import torch.nn as nn

from torch.nn.functional import softplus
from torch.distributions import Normal, Laplace, Gumbel, Uniform
from torch.nn.functional import relu as ReLU

t.manual_seed(0)
t.cuda.manual_seed(0)

VERBOSE = False

def identity(x):
    return x

def ReLU(x):
    return x * (x > 0)

def fit_approx_post(moments, dist_type=Normal):
    # moments is a vector nx2 
    # dist is a torch.distributions type/constructor: Normal, Laplace or Gumbel
    # returns a vector nx2 of parameters of the approximating distribution

    loc   = moments[:,0]
    raw_2nd_mom = moments[:,1] - loc**2
    bounded_2nd_mom = raw_2nd_mom + (-raw_2nd_mom + 1e-10)*(raw_2nd_mom<=0)
    scale = bounded_2nd_mom.sqrt()

    if dist_type == Laplace:
        # Variance of Laplace(mu, b) is 2*b^2
        scale /= 2**0.5
    elif dist_type == Gumbel:
        # Variance of Gumbel(mu, b) is (pi^2)/6 * b^2
        # Mean of Gumbel(mu, b) is mu + 0.57721...*b
        # (where 0.57721...W is the Euler-Mascheroni constant)
        scale *= math.sqrt(6)/math.pi
        loc -= 0.57721*scale

    params = t.vstack([loc, scale]).t()

    return params

def mean2nat(means):
    return conv2nat(fit_approx_post(means))


def conv2nat(convs):
    loc = convs[:,0]
    scale = convs[:,1]
    prec = 1/scale
    mu_prec = loc * prec
    nats = t.vstack([mu_prec, -0.5*prec]).t()
    return nats

def nat2conv(nats):
    mu_prec = nats[:,0]
    minus_half_prec = nats[:,1]
    prec = -2*minus_half_prec
    loc = mu_prec / prec
    scale = prec.rsqrt()
    convs = t.vstack([loc, scale]).t()
    return convs

def IW(sample, approx_dist, post=None, prior=None, likelihood=None, elf=0):
    # sample is a list of samples
    # params is a vector nx2
    # post is a torch distribution
    # dist is a torch.distributions type/constructor: Normal, Laplace or Gumbel
    # returns a list of IW samples and the ELBO

    if post is None and (prior is not None and likelihood is not None):
        logp = prior.log_prob(sample) + likelihood
    else:
        logp = post.log_prob(sample)

    logq = approx_dist.log_prob(sample)
    
    K = sample.shape[0]
    N = sample.shape[1]
    
    elbo = t.logsumexp((logp - logq + elf), dim=0).sum() - N*math.log(K)
    
    lqp = logp - logq
    lqp_max = lqp.amax(axis=0)
    weights = t.exp(lqp - lqp_max)

    weights = weights / weights.sum(axis=0)
    

    Ex_one_iter = (weights * sample).sum(axis=0)
    Ex2_one_iter = (weights * sample**2).sum(axis=0)
    m_one_iter = t.stack([Ex_one_iter, Ex2_one_iter]).t()

    return m_one_iter, elbo


def sample(dist, K=1):
    # params is a vector nx2
    # K is the number of samples to draw per latent variable
    # dist is a torch.distributions type/constructor: Normal, Laplace or Gumbel
    # returns a list of samples

    samps = dist.sample((K,))
    return samps

def entropy(dist):
    # params is a vector nx2
    # dist is a torch.distributions type/constructor: Normal, Laplace or Gumbel

    return dist.entropy().sum()

 
def ammp_is(T, post_params, init_moments, lr, K=5, approx_post_type=Normal, post_type=Normal, use_ReLU=False, num_inner_loop=1):
    m_q = [init_moments]
    m_avg = [init_moments]
    l_tot = [-1e15]
    l_one_iters = []
    log_w_t_minus_one = 0.0
    
    times = t.zeros(T+1)
    start_time = time.time()

    weights = [0]

    post = post_type(post_params[:,0], post_params[:,1])

    init_params = fit_approx_post(m_q[-1], approx_post_type)
    init_dist = approx_post_type(init_params[:,0], init_params[:,1])
    entropies = [entropy(init_dist)]
    for i in range(T):
        Q_params = fit_approx_post(m_q[-1], approx_post_type)

        Q_t = approx_post_type(Q_params[:,0], Q_params[:,1])
        z_t = sample(Q_t, K)

        m_one_iter_t, l_one_iter_t = IW(z_t, Q_t, post)

        H_Q = entropy(Q_t)
        H_Q_temp = entropy(Q_t)

        if VERBOSE and i % 100 == 0:
            print("Iteration: ", i, "ELBO: ", l_tot[-1])

        for _ in range(num_inner_loop):
            if use_ReLU:
                log_w_t = -ReLU(H_Q - H_Q_temp)
            else:
                log_w_t = -(H_Q - H_Q_temp)

            
            dt = log_w_t - log_w_t_minus_one #- 0.1

            l_tot_t = l_tot[-1] - dt + softplus(l_one_iter_t + dt - l_tot[-1])
            eta_t = np.exp(l_one_iter_t - l_tot_t)
          
            new_m_avg = eta_t * m_one_iter_t + (1 - eta_t) * m_avg[-1]

            Q_temp_params = fit_approx_post(new_m_avg, approx_post_type)


            Q_temp = approx_post_type(Q_temp_params[:,0], Q_temp_params[:,1])
            H_Q_temp = entropy(Q_temp)

        weights.append(log_w_t.clone())
        l_one_iters.append(l_one_iter_t.clone())
        l_tot.append(l_tot_t.clone())
        m_avg.append(new_m_avg.clone())

        m_q.append(lr * new_m_avg + (1 - lr) * m_q[-1])

        times[i+1] = time.time() - start_time
        entropies.append(H_Q)

        log_w_t_minus_one = log_w_t

    return m_q, m_avg, l_tot, l_one_iters, weights, entropies, times

def ammp_is_ReLU(T, post_params, init_moments, lr, K=5, approx_post_type=Normal, post_type=Normal, num_inner_loop=1):
    return ammp_is(T, post_params, init_moments, lr, K, approx_post_type, post_type, use_ReLU=True, num_inner_loop=num_inner_loop)

def ammp_is_uniform_dt(T, post_params, init_moments, lr, K=5, approx_post_type=Normal, post_type=Normal):
    m_q = [init_moments]
    m_avg = [init_moments]
    l_tot = [-1e15]
    l_one_iters = []
    times = t.zeros(T+1)
    start_time = time.time()

    post = post_type(post_params[:,0], post_params[:,1])

    init_params = fit_approx_post(m_q[-1], approx_post_type)
    init_dist = approx_post_type(init_params[:,0], init_params[:,1])
    entropies = [entropy(init_dist)]
    for i in range(T):
        Q_params = fit_approx_post(m_q[-1], approx_post_type)

        Q_t = approx_post_type(Q_params[:,0], Q_params[:,1])
        z_t = sample(Q_t, K)

        m_one_iter_t, l_one_iter_t = IW(z_t, Q_t, post)

        if VERBOSE and i % 100 == 0:
            print("Iteration: ", i, "ELBO: ", l_tot[-1])

        dt = 0.0

        l_tot_t = l_tot[-1] + dt + softplus(l_one_iter_t - dt - l_tot[-1])
        eta_t = np.exp(l_one_iter_t - l_tot_t)

        new_m_avg = eta_t * m_one_iter_t + (1 - eta_t) * m_avg[-1]


        l_tot.append(l_tot_t.clone())
        l_one_iters.append(l_one_iter_t.clone())
        m_avg.append(new_m_avg.clone())

        m_q.append(lr * new_m_avg + (1 - lr) * m_q[-1])

        times[i+1] = time.time() - start_time
        entropies.append(entropy(Q_t))

    return m_q, m_avg, l_tot, l_one_iters, [0]*len(l_tot), entropies, times

def ammp_is_no_inner_loop(T, post_params, init_moments, lr, K=5, approx_post_type=Normal, post_type=Normal, use_ReLU=False):
    m_q = [init_moments]
    m_avg = [init_moments]
    l_tot = [-1e15]
    l_one_iters = []
    weights = [0]
    times = t.zeros(T+1)
    start_time = time.time()

    post = post_type(post_params[:,0], post_params[:,1])

    Q_temp_params = fit_approx_post(m_q[-1], approx_post_type)
    Q_temp = approx_post_type(Q_temp_params[:,0], Q_temp_params[:,1])
    H_Q_temp = entropy(Q_temp)
    entropies = [H_Q_temp]
    for i in range(T):
        Q_params = fit_approx_post(m_q[-1], approx_post_type) 

        Q_t = approx_post_type(Q_params[:,0], Q_params[:,1])
        z_t = sample(Q_t, K)

        H_Q = entropy(Q_t)
        
        if VERBOSE and i % 100 == 0:
            print("Iteration: ", i, "ELBO: ", l_tot[-1])

        if use_ReLU:
            dt = -ReLU(H_Q - H_Q_temp)
        else:
            dt = -(H_Q - H_Q_temp) + 0.1

        m_one_iter_t, l_one_iter_t = IW(z_t, Q_t, post)  

        l_tot_t = l_tot[-1] + dt + softplus(l_one_iter_t - dt - l_tot[-1])
        eta_t = np.exp(l_one_iter_t - l_tot_t)
        
        new_m_avg = eta_t * m_one_iter_t + (1 - eta_t) * m_avg[-1]

        l_tot.append(l_tot_t.clone())
        l_one_iters.append(l_one_iter_t.clone())
        m_avg.append(new_m_avg.clone())

        m_q.append(lr * new_m_avg + (1 - lr) * m_q[-1])

        H_Q_temp = H_Q
        weights.append(dt)
        times[i+1] = time.time() - start_time
        entropies.append(H_Q)

    return m_q, m_avg, l_tot, l_one_iters, weights, entropies, times

def ammp_is_no_inner_loop_ReLU(T, post_params, init_moments, lr, K=5, approx_post_type=Normal, post_type=Normal):
    return ammp_is_no_inner_loop(T, post_params, init_moments, lr, K, approx_post_type, post_type, use_ReLU=True)


def ammp_is_weight_all(T, post_params, init_moments, lr, K=5, approx_post_type=Normal, post_type=Normal):
    m_q = [init_moments]
    m_one_iters = []
    l_one_iters = [-1e15]
    m_avg = [init_moments]
    l_tot = [-1e15]
    l_one_iters = []

    times = t.zeros(T+1)
    start_time = time.time()

    post = post_type(post_params[:,0], post_params[:,1])

    Q_params = fit_approx_post(m_q[-1], approx_post_type)
    Q_1 = approx_post_type(Q_params[:,0], Q_params[:,1])
    entropies = [entropy(Q_1)]
    for i in range(T):
        Q_params = fit_approx_post(m_q[-1], approx_post_type) 

        Q_t = approx_post_type(Q_params[:,0], Q_params[:,1])
        z_t = sample(Q_t, K)

        m_one_iter_t, l_one_iter_t = IW(z_t, Q_t, post)  

        m_one_iters.append(m_one_iter_t)
        l_one_iters.append(l_one_iter_t)
        if VERBOSE and i % 100 == 0:
            print("Iteration: ", i, "ELBO: ", l_tot[-1])

        params_t = [fit_approx_post(m, approx_post_type) for m in m_q]
        dts = [-ReLU(entropy(approx_post_type(p[:,0], p[:,-1])) - entropy(Q_t)) for p in params_t]
        lts = t.stack([lt - dt for lt, dt in zip(l_one_iters, dts)])

        l_tot.append(t.logsumexp(lts, dim=0).clone())
        # l_one_iters.append(l_one_iter_t.clone())

        new_m_avg = sum([t.exp(lt - l_tot[-1]) * m for lt, m in zip(lts, m_one_iters)])

        m_avg.append(new_m_avg)

        m_q.append(lr * new_m_avg + (1 - lr) * m_q[-1])

        times[i+1] = time.time() - start_time

        entropies.append(entropy(Q_t))

    return m_q, m_avg, l_tot, l_one_iters, [0] + dts, entropies, times

def natural_rws(T, init_moments, lr, K=5, prior_params=None, lik_params=None, post_params=None, approx_post_type=Normal, prior_type=Normal, like_type=Normal, data=None):
    # to allow for lr schedules, we'll define lr as a function of iteration number (i)
    if type(lr) == float:
        lr_fn = lambda i: lr
    else:
        lr_fn = lr

    m_q = [init_moments]
    l_one_iters = []
    
    times = t.zeros(T+1)
    start_time = time.time()

    if prior_params is not None and lik_params is not None:
        prior = prior_type(prior_params[:,0], prior_params[:,1])
    else:
        post = post_type(post_params[:,0], post_params[:,1])

    init_params = fit_approx_post(m_q[-1], approx_post_type)
    init_dist = approx_post_type(init_params[:,0], init_params[:,1])
    entropies = [entropy(init_dist)]
    for i in range(T):
        Q_params = fit_approx_post(m_q[-1], approx_post_type)

        Q_t = approx_post_type(Q_params[:,0], Q_params[:,1])

        z_t = sample(Q_t, K)
        if prior_params is not None and lik_params is not None:
            likelihood = like_type(z_t, lik_params).log_prob(data)
            m_one_iter_t, l_one_iter_t = IW(z_t, Q_t, prior = prior, likelihood = likelihood)
        else:
            m_one_iter_t, l_one_iter_t = IW(z_t, Q_t, post=post)

        new_m_q = lr_fn(i) * m_one_iter_t + (1 - lr_fn(i)) * m_q[-1]

        # input(f"{i}, {(new_m_q-m_q[-1]).abs().mean()}")


        entropies.append(entropy(Q_t))
        l_one_iters.append(l_one_iter_t.clone())
        m_q.append(new_m_q.clone())

        times[i+1] = time.time() - start_time

    return m_q, l_one_iters, entropies, times


def ml1(T, init_moments, lr, K=5, prior_params=None, lik_params=None, post_params=None, approx_post_type=Normal, prior_type=Normal, like_type=Normal, data=None):
    # to allow for lr schedules, we'll define lr as a function of iteration number (i)
    if type(lr) == float:
        lr_fn = lambda i: lr
    else:
        lr_fn = lr

    m_q = [init_moments.requires_grad_()]
    l_one_iters = []
    
    times = t.zeros(T+1)
    start_time = time.time()

    if prior_params is not None and lik_params is not None:
        prior = prior_type(prior_params[:,0], prior_params[:,1])
    else:
        post = post_type(post_params[:,0], post_params[:,1])

    init_params = fit_approx_post(m_q[-1], approx_post_type)
    init_dist = approx_post_type(init_params[:,0], init_params[:,1])
    entropies = [entropy(init_dist)]
    
    for i in range(T):
        nats = mean2nat(m_q[-1])
        nats.retain_grad()
        Q_params = nat2conv(nats)
        Q_t = approx_post_type(Q_params[:,0], Q_params[:,1])

        z_t = sample(Q_t, K)
        if prior_params is not None and lik_params is not None:
            likelihood = like_type(z_t, lik_params).log_prob(data)
            m_one_iter_t, l_one_iter_t = IW(z_t, Q_t, prior = prior, likelihood = likelihood)
        else:
            m_one_iter_t, l_one_iter_t = IW(z_t, Q_t, post=post)

        l_one_iter_t.backward()
        new_m_q = m_q[-1] + lr_fn(i) * nats.grad

        # input(f"{i}, {(new_m_q-m_q[-1]).abs().mean()}")


        entropies.append(entropy(Q_t))
        l_one_iters.append(l_one_iter_t.clone())
        m_q.append(new_m_q.clone())

        times[i+1] = time.time() - start_time

    for i in range(T):
        m_q[i] = m_q[i].detach()
        l_one_iters[i] = l_one_iters[i].detach()

    return m_q, l_one_iters, entropies, times

def ml2(T, init_moments, lr, K=5, prior_params=None, lik_params=None, post_params=None, approx_post_type=Normal, prior_type=Normal, like_type=Normal, data=None):
    # to allow for lr schedules, we'll define lr as a function of iteration number (i)
    if type(lr) == float:
        lr_fn = lambda i: lr
    else:
        lr_fn = lr

    m_q = [init_moments]
    l_one_iters = []
    
    times = t.zeros(T+1)
    start_time = time.time()

    if prior_params is not None and lik_params is not None:
        prior = prior_type(prior_params[:,0], prior_params[:,1])
    else:
        post = post_type(post_params[:,0], post_params[:,1])

    init_params = fit_approx_post(m_q[-1], approx_post_type)
    init_dist = approx_post_type(init_params[:,0], init_params[:,1])
    entropies = [entropy(init_dist)]
    for i in range(T):
        J_loc = t.zeros_like(init_params[:,0]).requires_grad_()
        J_scale = t.zeros_like(init_params[:,1]).requires_grad_()
        Q_params = fit_approx_post(m_q[-1], approx_post_type)

        Q_t = approx_post_type(Q_params[:,0], Q_params[:,1])

        z_t = sample(Q_t, K)



        elf = sum(sum(J*f(z_t) for J,f in zip((J_loc, J_scale),(identity, t.square))))

        if prior_params is not None and lik_params is not None:
            likelihood = like_type(z_t, lik_params).log_prob(data)
            m_one_iter_t, l_one_iter_t = IW(z_t, Q_t, prior = prior, likelihood = likelihood, elf = elf)
        else:
            m_one_iter_t, l_one_iter_t = IW(z_t, Q_t, post=post, extra_log_factor = elf)


        l_one_iter_t.backward()
        
        new_m_q = m_q[-1] * (1 - lr_fn(i)) + lr_fn(i) * t.vstack([J_loc.grad, J_scale.grad]).t()

        # input(f"{i}, {(new_m_q-m_q[-1]).abs().mean()}")


        entropies.append(entropy(Q_t))
        l_one_iters.append(l_one_iter_t.clone())
        m_q.append(new_m_q.clone())

        times[i+1] = time.time() - start_time

    return m_q, l_one_iters, entropies, times


def natural_rws_difference(T, post_params, init_moments, lr, K=5, approx_post_type=Normal, post_type=Normal):
    m_q = [init_moments]
    l_one_iters = []
    
    times = t.zeros(T+1)
    start_time = time.time()

    post = post_type(post_params[:,0], post_params[:,1])

    init_params = fit_approx_post(m_q[-1], approx_post_type)
    init_dist = approx_post_type(init_params[:,0], init_params[:,1])
    entropies = [entropy(init_dist)]
    for i in range(T):
        Q_params = fit_approx_post(m_q[-1], approx_post_type)

        Q_t = approx_post_type(Q_params[:,0], Q_params[:,1])

        z_t = sample(Q_t, K)

        m_one_iter_t, l_one_iter_t = IW(z_t, Q_t, post)

        Ex_one_iter = (z_t).sum(axis=0)
        Ex2_one_iter = (z_t**2).sum(axis=0)
        m_z = t.stack([Ex_one_iter, Ex2_one_iter]).t()

        new_m_q = lr * (m_one_iter_t - m_z) + m_q[-1]

        entropies.append(entropy(Q_t))
        l_one_iters.append(l_one_iter_t.clone())
        m_q.append(new_m_q.clone())

        times[i+1] = time.time() - start_time

    return m_q, l_one_iters, entropies, times

def natural_rws_standardised(T, post_params, init_moments, lr, K=5, approx_post_type=Normal, post_type=Normal):
    # to allow for lr schedules, we'll define lr as a function of iteration number (i)
    if type(lr) == float:
        lr_fn = lambda i: lr
    else:
        lr_fn = lr
        
    m_q = [init_moments]
    l_one_iters = []
    
    times = t.zeros(T+1)
    start_time = time.time()

    post = post_type(post_params[:,0], post_params[:,1])

    init_params = fit_approx_post(m_q[-1], approx_post_type)
    init_dist = approx_post_type(init_params[:,0], init_params[:,1])
    entropies = [entropy(init_dist)]
    for i in range(T):
        Q_params = fit_approx_post(m_q[-1], approx_post_type)

        Q_t = approx_post_type(Q_params[:,0], Q_params[:,1])

        z_t = sample(Q_t, K)

        mean_z = z_t.mean(axis=0)
        var_z = z_t.var(axis=0)
    
        z_t = np.sqrt(Q_params[:,1] / var_z) * (z_t - mean_z) + Q_params[:,0]

        m_one_iter_t, l_one_iter_t = IW(z_t, Q_t, post)

        new_m_q = lr_fn(i) * m_one_iter_t + (1 - lr_fn(i)) * m_q[-1]

        entropies.append(entropy(Q_t))
        l_one_iters.append(l_one_iter_t.clone())
        m_q.append(new_m_q.clone())

        times[i+1] = time.time() - start_time

    return m_q, l_one_iters, entropies, times


def VI(T, post_params, init_params, lr, approx_post_type=Normal, post_type=Normal, K=5):

    # approximate posterior
    means = nn.Parameter(t.tensor(init_params[:,0], dtype=t.float64), requires_grad=True)
    log_vars = nn.Parameter(t.tensor(init_params[:,1], dtype=t.float64), requires_grad=True)

    post = post_type(post_params[:,0], post_params[:,1])
    opt = t.optim.Adam([means,log_vars], lr=lr)

    mean_arr = []
    log_var_arr = []
    elbos = []
    entropies = []

    times = t.zeros(T+1)
    start_time = time.time()
    for i in range(T):
        opt.zero_grad()

        Q = approx_post_type(means, t.exp(log_vars))

        z = Q.rsample((K,))

        # compute ELBO
        K = z.shape[0]
        N = z.shape[1]
    
        elbo = t.logsumexp((post.log_prob(z)- Q.log_prob(z)), dim=0).sum() - N*math.log(K)

        # compute gradients
        (-elbo).backward()

        # update parameters
        opt.step()

        times[i+1] = time.time() - start_time

        if VERBOSE and i % 100 == 0:
            print("Iteration: ", i, "ELBO: ", elbo.item())
        
        mean_arr.append(means.clone().detach())
        log_var_arr.append(log_vars.clone().detach())
        elbos.append(elbo.item())
        entropies.append(entropy(Q).clone().detach())

    return mean_arr, log_var_arr, elbos, entropies, times

def HMC(T, post_params, init, post_type=Normal, num_chains=4):
    # NOTE: this produces a total of T*num_chains samples
    ll_stan_func = {Normal: "normal_lpdf", Laplace: "double_exponential_lpdf", Gumbel: "gumbel_lpdf"}
    code = """
    data {
        int<lower=1> N;
        array[N] real loc;
        array[N] real scale;
    }
    parameters {
        array[N] real y; 
    }
    model {
        target += %s(y | loc, exp(scale));
    }
    """ % (ll_stan_func[post_type])
    data = {"N": int(init.shape[0]), "loc": post_params[:,0].numpy(), "scale": t.log(post_params[:,1]).numpy()}
    # hmc_init = [{"loc": init[:,0].numpy(), "scale": t.log(init[:,1]).numpy()}]*num_chains
    hmc_init = [{"y": post_type(init[:,0], init[:,1]).sample((1,))[0,:].numpy()}]*num_chains

    posterior = stan.build(code, data=data)

    start_time = time.time()
    fit = posterior.sample(num_chains=num_chains, num_samples=T, init=hmc_init)
    end_time = time.time()

    times = np.arange(0, end_time - start_time, (end_time - start_time)/(T*num_chains))

    samples = fit["y"]

    moments = []

    for i in range(T*num_chains):
        # breakpoint()
        Ex = samples[:,:(i+1)].mean(1)
        Ex2 = (samples[:,:(i+1)]**2).mean(1)
        moments.append(t.stack([t.tensor(Ex), t.tensor(Ex2)]).t())

    return moments, times, samples

def mcmc(T, post_params, init, proposal_scale, burn_in=100, post_dist=Normal):
    if type(proposal_scale) in (float, int):
        proposal_scale = proposal_scale*t.ones(num_latents, 1)

    N = init.shape[0]

    init_proposal = Normal(init[:,0], init[:,1])
    x = sample(init_proposal, 1)
    
    # to store samples
    samples = t.zeros((T + burn_in, N))
    samples[0,:] = x

    moments = [init]

    num_accepted = t.zeros(N)

    times = t.zeros(T+1)
    start_time = time.time()

    post = post_dist(post_params[:,0], post_params[:,1])
    for i in range(T + burn_in):
        # normal proposal
        proposal_params = t.stack([x.squeeze(0), proposal_scale.squeeze(1)], dim=1)

        proposal = Normal(proposal_params[:,0], proposal_params[:,1])
        y = sample(proposal, 1)

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
            Ex = t.mean(samples[burn_in:(i+1),:], dim=0)
            Ex2 = t.mean(samples[burn_in:(i+1),:]**2, dim=0)
            moments.append(t.stack([Ex, Ex2]).t())

            times[i-burn_in+1] = time.time() - start_time

    acceptance_rates = num_accepted / (T + burn_in)

    return moments, acceptance_rates.mean().item(), times, samples

def lang(T, post_params, init, proposal_scale, burn_in=100, post_dist=Normal):
    if type(proposal_scale) in (float, int):
        proposal_scale = proposal_scale*t.ones(num_latents, 1)

    N = init.shape[0]

    init_proposal = Normal(init[:,0], init[:,1])
    x = sample(init_proposal, 1)
    
    # to store samples
    samples = t.zeros((T + burn_in, N))
    samples[0,:] = x

    moments = [init]

    num_accepted = t.zeros(N)

    times = t.zeros(T+1)
    start_time = time.time()

    post = post_dist(post_params[:,0], post_params[:,1])
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

        proposal = Normal(proposal_params[:,0], proposal_params[:,1])
        y = sample(proposal, 1)

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
            Ex = t.mean(samples[burn_in:(i+1),:], dim=0)
            Ex2 = t.mean(samples[burn_in:(i+1),:]**2, dim=0)
            moments.append(t.stack([Ex, Ex2]).t())

            times[i-burn_in+1] = time.time() - start_time

    acceptance_rates = num_accepted / (T + burn_in)

    return moments, acceptance_rates.mean().item(), times, samples

def get_errs(m_q, post_params):
    # m_q is a list of moments
    # post is a vector each row of which defines a posterior for a latent variable
    # returns a list of mean errors and a list of variance errors
    mean_errs = []
    var_errs = []

    for i in range(len(m_q)):
        mean_errs.append((post_params[:,0] - fit_approx_post(m_q[i])[:,0]).abs().mean())
        var_errs.append((post_params[:,1] - fit_approx_post(m_q[i])[:,1]).abs().mean())

    return mean_errs, var_errs


if __name__ == "__main__":
    num_latents = 500
    K=5
    init = t.tensor([0.0,1.0], dtype=t.float64).repeat((num_latents,1))

    loc = Normal(0,150).sample((num_latents,1)).float()
    # scale = Normal(0,1).sample((num_latents,1)).exp().float()
    scale = Uniform(-5,-0.5).sample((num_latents,1)).exp().float()

    # print(scale)
    # breakpoint()

    post_params = t.cat([loc, scale], dim=1)

    m_q, m_avg, l_tot, l_one_iters, log_weights, entropies, ammp_is_times = ammp_is(1000, post_params, init, 0.4, K)

    
        

