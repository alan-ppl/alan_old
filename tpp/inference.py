import torch as t
from .backend import *


def vi(logps, logqs):
    elbo = sum_logpqs(logps, logqs)
    return elbo

def reweighted_wake_sleep(logps, logqs):

    # ## Wake-phase Theta p update
    wake_theta_loss = sum_logpqs(logps, {n:lq.detach() for (n,lq) in logqs.items()})
    # print(wake_theta_loss)
    ## Wake-phase phi q update
    logps = {n:lp.detach() for (n,lp) in logps.items()}
    wake_phi_loss = sum_logpqs(logps, logqs)
    # print(wake_phi_loss)
    ## Sleep-phase phi q update

    return wake_theta_loss, wake_phi_loss






if __name__ == "__main__":
    a = t.randn(3,3).refine_names('K_d', 'K_b')
    ap = t.randn(3,3).refine_names('K_b', 'K_a')
    b = t.randn(3,3,3).refine_names('K_a', 'K_b', 'plate_s')
    c = t.randn(3,3,3).refine_names('K_c', 'K_d', 'plate_s')
    d = t.randn(3,3,3).refine_names('K_a', 'K_c', 'plate_b')
    lps = (a,b,c,d)

    assert t.allclose((a.exp() @ ap.exp()/3).log().rename(None), reduce_K([a, ap], 'K_b')[0].rename(None))

    lp, marginals = sum_lps(lps)

    # data = tpp.sample(P, "obs")
    # print(data)
    print(gibbs(marginals))
