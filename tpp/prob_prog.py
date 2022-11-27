import torch as t
import torch.distributions as td
import torch.nn as nn
from .wrapped_distribution import WrappedDist
from .tensor_utils import dename, get_dims, sum_none_dims
from functorch.dim import dims, Dim

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

    def __init__(self, data=None, dims=None, reparam=True, K_dim = None):
        super().__init__()
        if data is None:
            data = {}
        self.data = data
        self.sample = {}
        self.logp = {}
        self.reparam = reparam
        self.dims = dims
        self.K_dim = K_dim

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
        if self.reparam:
            sample = value.rsample(K=self.K_dim)
        else:
            sample = value.sample(K=self.K_dim)

        # # ## Assert no K being sampled, check value.sample_K
        # if not hasdim(self.dims['K'], get_dims(sample)):
        #     sample = sample#.unsqueeze(0)[self.dims['global']]

        self.sample[key] = sample
        # print(key)
        # print(sample)
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
        self.groups = {}

    def __getitem__(self, key):
        return self.sample[key]

    def __setitem__(self, key, value):
        assert isinstance(value, WrappedDist)
        assert key not in self.sample
        try:
            self.sample[key] = value.rsample()
            self.groups[key] = value.group
        except NotImplementedError:
            self.sample[key] = value.sample()
            self.groups[key] = value.group






class TraceLogP(Trace):
    def __init__(self, logq_trace, data=None, dims = None, K_dim = None):
        self.sample = logq_trace.sample
        self.logq = logq_trace.logp
        self.data = data
        # maintains an ordered list of tensors as they are generated
        self.logp = {}
        self.dims = {}
        self.K_dim = K_dim


    def __getitem__(self, key):
        # ensure tensor has been generated

        assert self.K_dim not in get_dims(self.sample)

        assert (key in self.data) or (key in self.sample)

        if key in self.sample:
            sample = self.sample[key]
            return sample

        return self.data[key]

    def __setitem__(self, key, value):
        assert isinstance(value, WrappedDist)
        assert (key in self.data) or (key in self.sample)

        group = value.group

        #rename p and q (log q and q samples) here
        if key in self.sample:
            K_name = 'K_{}'.format(group) if group is not None else f"K_{key}"
            if self.K_dim in get_dims(self.sample[key]):
                if key not in self.dims:
                    self.dims[key] = Dim(name=K_name, size=self.K_dim.size)
                self.sample[key] = self.sample[key].index(self.K_dim, self.dims[key])
                self.logq[key] = self.logq[key].rename(K=K_name)
                sample = self.sample[key]

            else:
                if key not in self.dims:
                    self.dims[key] = Dim(name=K_name, size=1)
                self.sample[key] = self.sample[key].unsqueeze(0)[self.dims[key]]
                self.logq[key] = self.logq[key].unsqueeze(0).refine_names(K_name)
                sample = self.sample[key]

        sample = self[key]
        self.logp[key] = value.log_prob(sample)
