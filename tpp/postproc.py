"""
A bunch of functions to do useful things with dicts of samples and weights.
"""
import torch as t
from .utils import *

def map_or_apply(f):
    def inner(sample_or_dict, w=None):
        if isinstance(sample_or_dict, dict):
            assert w is None
            return {varname: (f(*val) if isinstance(val, tuple) else f(val)) for (varname, val) in sample_or_dict.items()}
        else:
            return f(sample_or_dict, w)
    return inner

def Ef(f):
    def inner(sample, w=None):
        assert isinstance(sample, t.Tensor)
        if w is None:
            w = 1/sample.size('N')
        dim = 'N' if w is None else 'K'
        return (w.align_as(sample) * f(sample)).sum(dim)
    return map_or_apply(inner)

mean_raw     = lambda x: x
mean2_raw    = lambda x: x**2
p_lower_raw  = lambda value: lambda x: (x < value).to(dtype=x.dtype)
p_higher_raw = lambda value: lambda x: (x > value).to(dtype=x.dtype)

mean     = Ef(mean_raw)
mean2    = Ef(mean2_raw)
p_lower  = lambda value: Ef(p_lower_raw(value))
p_higher = lambda value: Ef(p_higher_raw(value))




"""
Relevant quantites that aren't plain moments
"""
@map_or_apply
def var(sample, w):
    return mean2(sample, w) - mean(sample, w)**2
@map_or_apply
def std(sample, w): 
    return var(sample, w).sqrt()
@map_or_apply
def ess(sample, w=None):
    assert isinstance(sample, t.Tensor)
    if w is None:
        raise Exception("ESS only makes sense for importance weights.  But here you're trying to compute ESS on just samples.")
    return 1/((w**2).sum('K'))

"""
Standard errors (which only make sense for plain moments)
Not subtracting 1 from the ESS?
"""
def stderr(f):
    return map_or_apply(lambda sample, w: (var(f(sample), w) / ess(sample, w)).sqrt())

stderr_mean     = stderr(mean_raw)
stderr_mean2    = stderr(mean2_raw)
stderr_p_lower  = lambda value: stderr(p_lower_raw(value))
stderr_p_higher = lambda value: stderr(p_higher_raw(value))
