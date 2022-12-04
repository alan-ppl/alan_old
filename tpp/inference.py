import torch as t
from .backend import *



def logPtmc(logps, logqs, val=None):
    """
    Arguments:
        logps: dict{rv_name -> log-probability tensor}
        logqs: dict{rv_name -> log-probability tensor}
    Returns:
        elbo, used for VI
    """
    all_tensors = combine_lpqs(logps, logqs)
    if val is not None:
        all_tensors.append(val)
    return sum_tensors(all_tensors)

def combine_lpqs(logps, logqs):
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

    for (rv, lq) in logqs.items():
        #only one k dimension in q
        assert len(tuple(name for name in lq.names if is_K(name)))
        #that k dimension is "K_$rv" where rv is the name of the random variable
        Kname = next(name for name in lq.names if is_K(name))
        assert Kname == K_prefix + rv

    # sanity checking for latents (only latents appear in logqs)
    for rv in logqs:
        #check that any rv in logqs is also in logps
        assert rv in logps

        lp = logps[rv]
        lq = logqs[rv]

        # check same plates/timeseries appear in lp and lq
        lp_notK = [n for n in lp.names if not is_K(n)]
        lq_notK = [n for n in lq.names if not is_K(n)]
        assert set(lp_notK) == set(lq_notK)

    #combine all lps, negating logqs
    all_lps = list(logps.values()) + [-lq for lq in logqs.values()]
    return all_lps






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
