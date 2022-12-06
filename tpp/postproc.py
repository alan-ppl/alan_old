"""
A bunch of functions to do useful things with dicts of samples and weights.
"""
import torch as t
from .backend import is_K

def Kname(x):
    Ks = tuple(K for K in x.names if is_K(K))
    assert 1==len(Ks)
    return Ks[0]

def map_or_apply(f):
    def inner(arg0, arg1=None):
        if isinstance(arg0, dict):
            assert arg1 is None
            return {var_name: f(sample, w) for (var_name, (sample, w)) in arg0.items()}
        else:
            assert arg1 is not None
            return f(arg0, arg1)
    return inner

def Ef(f):
    return map_or_apply(lambda sample, w: (w*f(sample)).sum(Kname(sample)))
mean_raw     = lambda x: x
mean2_raw    = lambda x: x**2
p_lower_raw  = lambda value: lambda x: (x < value).to(dtype=x.dtype)
p_higher_raw = lambda value: lambda x: (x > value).to(dtype=x.dtype)

mean     = Ef(mean_raw)
mean2    = Ef(mean2_raw)
p_lower  = lambda value: Ef(p_lower_raw(value))
p_higher = lambda value: Ef(p_lower_raw(value))



"""
Relevant quantites that aren't plain moments
"""
var = map_or_apply(lambda sample, w: mean2(sample, w) - (mean(sample, w))**2)
std = map_or_apply(lambda sample, w: var(sample, w).sqrt())
ess = map_or_apply(lambda _, w: 1/((w**2).sum(Kname(w))))

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
