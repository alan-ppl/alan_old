import torch as t
import torch.nn as nn
import torch.distributions as td

"""
Basic principle:
Write down P and Q as probabilistic programs (functions or PyTorch modules that take a trace as input).
These should be "compatible": if I sample from P and Q, I get tensors with same dimensions + names.
Note that this means we need a single K dimension if we sample from P.



"""


#### Wrapping dists so that they propagate named dimensions correctly

def unify_names(*nss):
    result = sum(t.zeros(len(ns)*(0,), names=ns) for ns in nss)
    if isinstance(result, t.Tensor):
        return result.names
    else:
        return ()

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
    def __init__(self, dist, *args, sample_shape=(), sample_names=()):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        if isinstance(sample_names, str):
            sample_names = (sample_names,)

        if sample_names==() and sample_shape!=():
            sample_names = len(sample_shape) * (None,)

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




class TraceSampleLogP():
    """
    Samples a probabilistic program + evaluates log-probability.
    Usually used for sampling the approximate posterior.
    Doesn't do any tensorisation.  All dimensions (e.g. sampling K) is managed by the user's program.
    Note that the latents may depend on the data (as in a VAE), but it doesn't make sense to "sample" data.
    """
    def __init__(self, K, data=None):
        self.K = K
        if data is None:
            data = {}
        self.data = data
        self.sample = {}
        self.logp = {}
    
    def __getitem__(self, key):
        if key in self.sample:
            assert key not in self.data
            return self.sample[key]
        else:
            assert key in self.data
            return self.data[key]

    def __setitem__(self, key, value):
        assert isinstance(value, WrappedDist)
        assert key not in self.data
        assert key not in self.sample
        sample = value.rsample()
        self.sample[key] = sample
        self.logp[key] = value.log_prob(sample)

class TraceSample():
    """
    Used for testing (e.g. to sample from the prior).
    Doesn't make sense to have data.
    No tensorisation.
    """
    def __init__(self, K):
        self.K = K
        self.sample = {}
    
    def __getitem__(self, key):
        return self.sample[key]

    def __setitem__(self, key, value):
        assert isinstance(value, WrappedDist)
        assert key not in self.sample
        self.sample[key] = value.rsample()

class StaticTensorisedTrace():
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

    def __setitem__(self, key, value):
        assert isinstance(value, WrappedDist)
        assert key not in data
        assert key not in sample
        sample = value.rsample()
        self.sample[key] = sample
        self.logp[key] = value.log_prob(sample)

    
def P(tr): 
    tr['a'] = Normal(0, 1, sample_shape=tr.K, sample_names="K")
    tr['b'] = Normal(tr['a'], 1)
    tr['c'] = Normal(tr['b'], 1, sample_shape=3, sample_names='plate_a')
    tr['obs'] = Normal(tr['c'], 1, sample_shape=5, sample_names='plate_b')


class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_a = nn.Parameter(t.zeros(()))
        self.m_b = nn.Parameter(t.zeros(()))
        self.m_c = nn.Parameter(t.zeros((3, 1)))

    def forward(self, tr):
        tr['a'] = Normal(self.m_a, 1, sample_shape=tr.K, sample_names='K')
        tr['b'] = Normal(self.m_b, 1, sample_shape=tr.K, sample_names='K')
        tr['c'] = Normal(self.m_c, 1, sample_shape=(3, sample_names='plate_a')

#sample fake data
tr_sample = TraceSample(K=1)
P(tr_sample)
data = {'obs': tr_sample.sample['obs']}

#sample from approximate posterior
trq = TraceSampleLogP(K=10, data=data)
q = Q()
q(trq)

#compute logP
tr = StaticTensorisedTrace(tr.sample, data)
test_prog_gen(tr)
