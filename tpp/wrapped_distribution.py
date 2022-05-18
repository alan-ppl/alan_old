import torch as t
import torch.distributions as td
from .cartesian_tensor import cartesiantensormap, tensormap, pad_nones, tensors


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

    def rsample(self):
        args = self.args
        kwargs = self.kwargs
        args, kwargs = cartesiantensormap(lambda x: x._t, args, kwargs)
        # Sorted list of all unique names
        unified_names = set([name for arg in tensors(args, kwargs) for name in arg.names])
        unified_names.discard(None)
        unified_names = sorted(unified_names)
        # Align tensors onto that sorted list
        args, kwargs = tensormap(lambda x: x.align_to(*unified_names, ...), args, kwargs)
        max_pos_dim = max(
            sum(name is None for name in arg.names) for arg in tensors(args, kwargs)
        )
        args, kwargs = tensormap(lambda x: pad_nones(x, max_pos_dim), args, kwargs)
        args, kwargs = tensormap(lambda x: x.rename(None), args, kwargs)
        return (self.dist(*args, **kwargs)
                .rsample(sample_shape=self.sample_shape)
                .refine_names(*self.sample_names, *unified_names, ...))


    def log_prob(self, x):
        assert isinstance(x, t.Tensor)
        args = (*self.args, x)
        kwargs = self.kwargs

        args, kwargs = cartesiantensormap(lambda x: x._t, args, kwargs)
        # Sorted list of all unique names
        unified_names = set([name for arg in tensors(args, kwargs) for name in arg.names])
        unified_names.discard(None)
        unified_names = sorted(unified_names)
        # Align tensors onto that sorted list
        args, kwargs = tensormap(lambda x: x.align_to(*unified_names, ...), args, kwargs)
        max_pos_dim = max(
            sum(name is None for name in arg.names) for arg in tensors(args, kwargs)
        )
        args, kwargs = tensormap(lambda x: pad_nones(x, max_pos_dim), args, kwargs)
        args, kwargs = tensormap(lambda x: x.rename(None), args, kwargs)
        return (self.dist(*args[:-1], **kwargs)
                .log_prob(args[-1])
                .refine_names(*unified_names, ...))

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
    "Weibull",
    "TransformedDistribution"
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
