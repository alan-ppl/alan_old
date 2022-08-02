import torch as t
import torch.distributions as td
import torch.nn as nn
from .cartesian_tensor import CartesianTensor
from .wrapped_distribution import WrappedDist
from .utils import *
# from torchdim import dims

__all__ = [
    'Trace', 'TraceSampleLogQ', 'TraceSample', 'TraceLogP'
]

class Trace:
    def __init__(self):
        """
        Initialize all Trace objects with K_shape and K_names.
        These should form the rightmost dimensions in all tensors in the program.
        """
        # assert len(K_shape) == len(K_names)

    def dist(self, dist, *args, **kwargs):
        """
        Check arg and kwarg are compatible before passing them to WrappedDist
        """
        return WrappedDist(dist, *args, **kwargs)

    def log_prob(self):
        return {
            k: v for (k, v) in self.logp.items()
        }


class TraceSampleLogQ(Trace):
    """
    Samples a probabilistic program + evaluates log-probability.
    Usually used for sampling the approximate posterior.
    The latents may depend on the data (as in a VAE), but it doesn't make sense to "sample" data.
    Can high-level latents depend on plated lower-layer latents?  (I think so?)
    """

    def __init__(self, data=None, dims=None):
        super().__init__()
        if data is None:
            data = {}
        self.data = data
        self.sample = {}
        self.logp = {}
        self.K = dims['K']

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
        sample = value.rsample(K=self.K)
        self.sample[key] = sample
        self.logp[key] = value.log_prob(sample)

    def __repr__(self) -> str:
        trace_repr = ''
        for name, value in self.sample.items():
            trace_repr += ("site: {}, shape: {}, name: {}\n".format(name,
                                                                    str(value.shape), value.names))
        return trace_repr


class TraceSample(Trace):
    """
    Just used e.g. to sample fake-data.  So no K's.
    """

    def __init__(self):
        super().__init__()
        self.sample = {}

    def __getitem__(self, key):
        return self.sample[key]

    def __setitem__(self, key, value):
        assert isinstance(value, WrappedDist)
        assert key not in self.sample
        self.sample[key] = value.rsample()





class TraceLogP(Trace):
    def __init__(self, sample, data=None, dims = None):
        self.sample = sample
        self.data = data
        # maintains an ordered list of tensors as they are generated
        self.logp = {}
        self.dims = dims


    def __getitem__(self, key):
        # ensure tensor has been generated
        assert (key in self.data) or (key in self.sample)
        if key in self.sample:
            K_name = f"K_{key}"
            # sample = self.sample[key].index(self.dims['K'], self.dims[K_name])
            sample = self.sample[key].rename(K=f"K_{key}")
            return sample
        return self.data[key]

    def __setitem__(self, key, value):
        assert isinstance(value, WrappedDist)
        assert (key in self.data) or (key in self.sample)
        sample = self[key]
        self.logp[key] = value.log_prob(sample)
