import torch as t
from .backend import *


def vi(logps, logqs):
    elbo = logPtmc(logps, logqs)
    return elbo

def reweighted_wake_sleep(logps, logqs):

    # ## Wake-phase Theta p update
    wake_theta_loss = logPtmc(logps, {n:lq.detach() for (n,lq) in logqs.items()})
    # print(wake_theta_loss)
    ## Wake-phase phi q update
    logps = {n:lp.detach() for (n,lp) in logps.items()}
    wake_phi_loss = logPtmc(logps, logqs)
    # print(wake_phi_loss)
    ## Sleep-phase phi q update

    return wake_theta_loss, wake_phi_loss

def combine_lps(logps, logqs):
    """
    Arguments:
        logps: dict{rv_name -> log-probability tensor}
        logqs: dict{rv_name -> log-probability tensor}
    Returns:
        all_lps: list of [*logps,  *(-logqs)]
    """
    #Data tensors appear in logps but not logqs
    assert len(logqs) <= len(logps)

    # check all dimensions are named, and are either plates or named K-dimensions
    for lp in [*logps.values(), *logqs.values()]:
        for n in lp.names:
            assert is_K(n) or is_plate(n)

    # sanity checking for latents (only latents appear in logqs)
    for rv in logqs:
        #check that any rv in logqs is also in logps
        assert rv in logps

        lp = logps[rv]
        lq = logqs[rv]

        # check same plates appear in lp and lq
        lp_plates = [n for n in lp.names if is_plate(n)]
        lq_plates = [n for n in lq.names if is_plate(n)]
        assert set(lp_plates) == set(lq_plates)

    #combine all lps, negating logqs
    all_lps = list(logps.values()) + [-lq for lq in logqs.values()]
    return all_lps

def logPtmc(logps, logqs):
    """
    Arguments:
        logps: dict{rv_name -> log-probability tensor}
        logqs: dict{rv_name -> log-probability tensor}
    Returns:
        elbo, used for VI
    """
    all_lps = combine_lps(logps, logqs)
    return combine_tensors(all_lps)





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
