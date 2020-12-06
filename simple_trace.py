import torch as t
import torch.distributions as td


#### Wrapping dists so that they propagate named dimensions correctly

def unify_names(*nss):
    result = sum(t.zeros(len(ns)*(0,), names=ns) for ns in nss)
    return result.names

def unify_arg_names(*args):
    return unify_names(*(arg.names for arg in args if isinstance(arg, t.Tensor)))

def strip_name(arg):
    if isinstance(arg, t.Tensor):
        return arg.rename(None)
    else:
        return arg

def strip_names(*args):
    return (strip_name(arg) for arg in args)


class WrappedDist:
    def __init__(self, dist, *args, sample_shape=t.Size([]), sample_names=[]):
        self.dist = dist(*strip_names(*args))
        self.unified_names = unify_arg_names(*args)
        self.sample_shape = sample_shape
        self.sample_names = sample_names
    
    def rsample(self):
        return self.dist.rsample(sample_shape=self.sample_shape) \
                .refine_names(*self.sample_names, *self.unified_names)
    
    def log_prob(self, x):
        return self.dist.log_prob(x) \
                .refine_names(*unify_names(x.names, self.unified_names))

dist_names = [
    "Bernoulli",
    "Beta",
    "Binomial",
    "Categorical",
    "Cauchy",
    "Chi2",
    "ContinuousBernoulli",
    "Exponential",
    "FisherSnedecor",
    "Gamma",
    "Geometric",
    "Gumbel",
    "HalfCauchy",
    "HalfNormal",
    "Laplace",
    "LogNormal",
    "NegativeBinomial",
    "Normal",
    "Pareto",
    "Poisson",
    "RelaxedBernoulli",
    "StudentT",
    "Uniform",
    "VonMises",
    "Weibull",
]

#Multivariate!
#Dirichlet
#MvNormal
#Multinomial
#OneHotCategorical
#RelaxedOneHotCategorical

def set_dist(dist_name):
    dist = getattr(td, dist_name)
    def inner(*args, **kwargs):
        return WrappedDist(dist, *args, **kwargs)
    globals()[dist_name] = inner

for dist_name in dist_names:
    set_dist(dist_name)




class TraceSampleK():
    """
    Used for sampling the approximate posterior.
    """
    def __init__(self, K, data=None):
        if data is None:
            data = {}
        self.data = data
        self.sample = {}
        self.logp = {}
    
    def __getitem__(self, key):
        if key in self.sample:
            assert key not in self.sample
            return self.sample[key]
        else:
            assert key in self.data
            return self.data[key]

    def __setindex__(self, key, value):
        assert isinstance(value, WrappedDist)
        assert key not in data
        assert key not in sample
        sample = value.rsample()
        self.sample[key] = sample
        self.logp[key] = value.log_prob(sample)

class TraceLogP():
    def __init__(self, sample, data):
        self.sample = sample
        self.data = data
        #maintains an ordered list of tensors as they are generated
        self.logp = {}

    def __getitem__(self, key):
        #ensure tensor has been generated
        assert key in self.logp

        if key in self.sample:
            assert key not in self.sample
            return self.sample[key]
        else:
            assert key in self.data
            return self.data[key]

    def __setindex__(self, key, value):
        assert isinstance(value, WrappedDist)
        assert key not in data
        assert key not in sample
        sample = value.rsample()
        self.sample[key] = sample
        self.logp[key] = value.log_prob(sample)

    
 

#Basic syntax:
#tr["a"] = Normal(..., sample_shape=(4,), sample_names="K")
#tr["b"] = Normal(tr["a"], 1)
