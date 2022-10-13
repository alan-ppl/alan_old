import torch as t
import torch.distributions as td
from .tensor_utils import *
from functorch.dim import Tensor as DimTensor
from functorch.dim import Dim

class WrappedDist:
    """
    A wrapper of torch.distributions that supports named tensor.
    """
    def __init__(self, dist, *args, sample_dim=(), sample_K=True, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.dist = dist

        if isinstance(sample_dim, Dim):
            self.sample_dim = sample_dim
            sample_shape = (sample_dim.size,)
            sample_dim = (sample_dim,)
        if isinstance(sample_dim, int):
            sample_dim = (sample_dim,)
        if sample_dim == () :
            self.sample_dim = None
            sample_shape = ()

        self.sample_K = sample_K
        self.sample_shape = sample_shape

        self.sample_names = tuple([repr(dim) for dim in sample_dim])

    def rsample(self, K=None):
        args = self.args
        kwargs = self.kwargs
        K_size = K.size if K is not None else None
        args, kwargs, denamify = nameify(args, kwargs)
        # print('sample_shape')
        # print(self.sample_shape)
        # print(self.sample_names)
        # Sorted list of all unique names
        unified_names = set([name for arg in tensors(args, kwargs) for name in arg.names])
        unified_names.discard(None)
        unified_names = sorted(unified_names)
        # print(unified_names)
        # for a in args:
        #     print(a.shape)
        already_K = 'K' in unified_names

        sampling_K = K_size is not None and not already_K and self.sample_K



        # sample_shape = (*self.sample_shape, K_size) if sampling_K else self.sample_shape
        sample_shape = self.sample_shape + (K_size,) if sampling_K else self.sample_shape

        # if not self.sample_K:
        #     sample_shape = sample_shape + (1,)
        #     print('sample shape')
        #     print(sample_shape)
        #Checking the user hasn't mistakenely labelled two variables with the same plate name
        if len(list(unified_names)) > 0:
            assert list(unified_names) != list(self.sample_names), "Don't label two variables with the same plate, it is unneccesary!"


        # # Align tensors onto that sorted list
        args, kwargs = tensormap(lambda x: x.align_to(*unified_names, ...), args, kwargs)
        max_pos_dim = max(
            sum(name is None for name in arg.names) for arg in tensors(args, kwargs)
        )
        # print(max_pos_dim)
        # for a in args:
        #     print(a.names)
        #Wargs, kwargs = tensormap(lambda x: pad_nones(x, max_pos_dim), args, kwargs)
        args, kwargs = tensormap(lambda x: x.rename(None), args, kwargs)
        # for a in args:
        #     print(a.shape)
        unified_names = ['K'] + sorted(unified_names) if sampling_K else sorted(unified_names)

        # if not self.sample_K:
        #     unified_names = [name] + sorted(unified_names)

        samples = (self.dist(*args, **kwargs)
                .rsample(sample_shape=sample_shape)
                .refine_names(*self.sample_names, *unified_names, ...))

        return denamify(samples, sample_dims = self.sample_dim, K_dim = K)

    def sample(self, K=None):
        args = self.args
        kwargs = self.kwargs
        K_size = K.size if K is not None else None
        args, kwargs, denamify = nameify(args, kwargs)
        # print('sample_shape')
        # print(self.sample_shape)
        # print(self.sample_names)
        # Sorted list of all unique names
        unified_names = set([name for arg in tensors(args, kwargs) for name in arg.names])
        unified_names.discard(None)
        unified_names = sorted(unified_names)
        # print(unified_names)
        # for a in args:
        #     print(a.shape)
        already_K = 'K' in unified_names
        sampling_K = K_size is not None and not already_K


        # sample_shape = (*self.sample_shape, K_size) if sampling_K else self.sample_shape
        sample_shape = self.sample_shape + (K_size,) if sampling_K else self.sample_shape


        #Checking the user hasn't mistakenely labelled two variables with the same plate name
        if len(list(unified_names)) > 0:
            assert list(unified_names) != list(self.sample_names), "Don't label two variables with the same plate, it is unneccesary!"


        # # Align tensors onto that sorted list
        args, kwargs = tensormap(lambda x: x.align_to(*unified_names, ...), args, kwargs)
        max_pos_dim = max(
            sum(name is None for name in arg.names) for arg in tensors(args, kwargs)
        )
        # print(max_pos_dim)
        # for a in args:
        #     print(a.names)
        #Wargs, kwargs = tensormap(lambda x: pad_nones(x, max_pos_dim), args, kwargs)
        args, kwargs = tensormap(lambda x: x.rename(None), args, kwargs)
        # for a in args:
        #     print(a.shape)
        unified_names = ['K'] + sorted(unified_names) if sampling_K else sorted(unified_names)
        # print('sample_shape')
        # print(sample_shape)
        # print(self.sample_names)
        samples = (self.dist(*args, **kwargs)
                .sample(sample_shape=sample_shape)
                .refine_names(*self.sample_names, *unified_names, ...))

        samples = samples.detach()
        return denamify(samples, sample_dims = self.sample_dim, K_dim = K)


    def log_prob(self, x):
        assert isinstance(x, t.Tensor) or isinstance(x, DimTensor)
        args = (*self.args, x)
        kwargs = self.kwargs

        # args, kwargs = cartesiantensormap(lambda x: x._t, args, kwargs)
        args, kwargs, denamify = nameify(args, kwargs)

        # Sorted list of all unique names
        unified_names = set([name for arg in tensors(args, kwargs) for name in arg.names])
        unified_names.discard(None)
        unified_names = sorted(unified_names)
        # Align tensors onto that sorted list
        args, kwargs = tensormap(lambda x: x.align_to(*unified_names, ...), args, kwargs)
        max_pos_dim = max(
            sum(name is None for name in arg.names) for arg in tensors(args, kwargs)
        )

        #args, kwargs = tensormap(lambda x: pad_nones(x, max_pos_dim), args, kwargs)
        args, kwargs = tensormap(lambda x: x.rename(None), args, kwargs)
        # print('after pad nones')
        # for a in args[:-1]:
        #     print(a.shape)
        log_probs = (self.dist(*args[:-1], **kwargs)
                .log_prob(args[-1])
                .refine_names(*unified_names, ...))


        return log_probs

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
    "Multinomial",
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
