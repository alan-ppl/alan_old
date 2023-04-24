import torch as t

def dictmap(f, d):
    return {varname: f(val) for (varname, val) in d.items()}

def map_dict_or_val(f):
    """
    Converts a function that just takes a tensor/tuple to a function that could
    take a tensor/tuple (and does the same thing) or could take a dict, and maps
    the function over a dictionary.
    """
    def inner(sample_or_dict):
        if isinstance(sample_or_dict, dict):
            return dictmap(f, sample_or_dict)
        else:
            return f(sample_or_dict)
    return inner

def map_optionally_weighted_sample(f):
    """
    Takes a function that applies only to a t.Tensor, and converts it to a function
    that could take a t.Tensor (and does the same thing), or could take a two-element
    tuple, and applies the function to the first element of the tuple (the sample)
    while keeping the second element of the tuple (the weights) the same.
    """
    def inner(x):
        if isinstance(x, tuple):
            sample, w = x
            return (f(sample), w)
        else:
            return f(x)
    return inner

def map_everything(f):
    return map_dict_or_val(map_optionally_weighted_sample(f))

"""
Functions that "map" over the samples.
"""
@map_everything
def identity(sample):
    return sample
@map_everything
def square(sample):
    return sample**2
@map_everything
def log(sample):
    return t.log(sample)
lower  = lambda value: map_everything(lambda x: (x < value).to(dtype=x.dtype))
higher = lambda value: map_everything(lambda x: (x > value).to(dtype=x.dtype))


"""
Functions that "reduce" over multiple (weighted) samples
"""

@map_dict_or_val
def mean(x):
    """
    Takes the mean.  Can handle a dict, and can handle a single tensor, or
    a tuple of a sample and a weight.
    """
    if isinstance(x, tuple):
        sample, w = x
        assert [name for name in sample.names if name is not None] == [name for name in w.names if name is not None]
        w = w.align_as(sample)
        dim = 'K'
    else:
        sample = x
        dim = 'N'
        w = 1/sample.size(dim)
    return (w * sample).sum(dim)

@map_dict_or_val
def var(x):
    return mean(square(x)) - square(mean(x))
@map_dict_or_val
def std(x):
    return var(x).sqrt()
@map_dict_or_val
def stderr(x):
    return (var(x) / ess(x)).sqrt()

@map_dict_or_val
def ess(x):
    if not isinstance(x, tuple):
        raise Exception(
            "Trying to compute the ESS for an unweighted sample."
            "It only makes sense to compute the ESS for a weighted sample"
        )
    _, w = x
    return 1/((w**2).sum('K'))
