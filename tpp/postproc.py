"""
A bunch of functions to do useful things with dicts of samples and weights.
"""
import torch as t
from .backend import is_K

def Kname(x):
    Ks = tuple(K for K in x.names if is_K(K))
    assert 1==len(Ks)
    return Ks[0]

def map_sample_dict(f):
    def inner(d):
        return {var_name: f(sample, w) for (var_name, (sample, w)) in d.items()}
    return inner

def mean_inner(sample, w):
    return (w*sample).sum(Kname(sample))

def var_inner(sample, w):
    K = Kname(sample)
    mean  = (w*sample   ).sum(K)
    mean2 = (w*sample**2).sum(K)
    return mean2 - mean**2

def std_inner(sample, w):
    return var_inner(sample, w).sqrt()

def ess_inner(_, w):
    return 1/((w**2).sum(Kname(w)))

mean = map_sample_dict(mean_inner)
var  = map_sample_dict(var_inner)
std  = map_sample_dict(std_inner)
ess  = map_sample_dict(ess_inner)


