import torch as t

def dictmap(f, d):
    return {varname: f(val) for (varname, val) in d.items()}

def map_or_apply(f):
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

def opt_weights(f):
    """
    Takes a function that applies only to a t.Tensor, and converts it to a function
    that could take a t.Tensor (and does the same thing), or could take a two-element
    tuple, and applies the function to the first element of the tuple (the sample)
    while keeping the second element of the tuple (the weights) the same.
    """
    def inner(sample_and_opt_weights):
        if isinstance(sample_and_opt_weights, tuple):
            assert 2==len(sample_and_opt_weights)
            w = sample_and_opt_weights[1]
            sample = sample_and_opt_weights[0]
            return (f(sample), w)
        else:
            return f(sample_and_opt_weights)
    return inner

def map_or_apply_opt_weights(f):
    return map_or_apply(opt_weights(f))

@map_or_apply_opt_weights
def identity(sample):
    return sample
@map_or_apply_opt_weights
def square(sample):
    return sample**2
@map_or_apply_opt_weights
def log(sample):
    return t.log(sample)
lower  = lambda value: map_or_apply_opt_weights(lambda x: (x < value).to(dtype=x.dtype))
higher = lambda value: map_or_apply_opt_weights(lambda x: (x > value).to(dtype=x.dtype))

@map_or_apply
def mean(sample_and_opt_weights):
    if isinstance(sample_and_opt_weights, tuple):
        assert 2==len(sample_and_opt_weights)
        w = sample_and_opt_weights[1]
        sample = sample_and_opt_weights[0]
    else:
        sample = sample_and_opt_weights
        w = 1/sample.size('N')
    dim = 'N' if (w is None) else 'K'
    return (w.align_as(sample) * sample).sum(dim)

@map_or_apply
def var(x):
    return mean(square(x)) - square(mean(x))
@map_or_apply
def std(x):
    return var(x).sqrt()
@map_or_apply
def stderr(x):
    return (var(x) / ess(x)).sqrt()

@map_or_apply
def ess(sample_and_opt_weights):
    assert isinstance(sample_and_opt_weights, tuple)
    assert 2==len(sample_and_opt_weights)
    w = sample_and_opt_weights[1]
    return 1/((w**2).sum('K'))
