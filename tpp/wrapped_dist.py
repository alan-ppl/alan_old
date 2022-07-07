import torch as t
import torch.distributions as td
from utils import *

class WrappedDist:
    """
    A wrapper of torch.distributions that supports named tensor.
    """
    def __init__(self, dist, *args, sample_shape=(), sample_names=(), **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.dist = dist

        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        if isinstance(sample_names, str):
            sample_names = (sample_names,)
        if sample_names == () and sample_shape != ():
            sample_names = len(sample_shape) * (None,)

        self.sample_shape = sample_shape
        self.sample_names = sample_names

    def rsample(self, K=None):
        names = get_names(self.args)
        args = dename(self.args, K is not None)
        dims = get_sizes(self.args[0])
        kwargs = self.kwargs

        if K is not None:
            sample_shape = (K.size,) + self.sample_shape
            sample_names = (K, names[0]) + self.sample_names
        else:
            sample_shape = self.sample_shape
            sample_names = (names[0]) + self.sample_names

        return (self.dist(*args, **kwargs)
                .rsample(sample_shape=sample_shape)[sample_names])


    def log_prob(self, x):
        args = (*self.args, x)
        names = get_names([x])
        args = dename(args, False)
        dims = get_sizes(self.args[0])
        kwargs = self.kwargs

        sample_names = (names[0]) + self.sample_names
        return (self.dist(*args[:-1], **kwargs)
                .log_prob(args[-1])[sample_names])

# Some distributions do not have rsample, how to handle? (e.g. Bernoulli)
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
    "MultivariateNormal",
    "NegativeBinomial",
    "Normal",
    "Pareto",
    "Poisson",
    "RelaxedBernoulli",
    "StudentT",
    "Uniform",
    "VonMises",
    "Weibull"
]
__all__ = dist_names

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
