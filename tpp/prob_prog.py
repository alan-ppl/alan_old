import torch as t
import torch.distributions as td
import torch.nn as nn
from .wrapped_distribution import WrappedDist
from .tensor_utils import dename, get_dims, sum_none_dims, has_dim
from .backend import is_K, is_plate

from functorch.dim import dims, Dim
__all__ = [
    'Trace', 'TraceSampleLogQ', 'TraceSample', 'TraceLogP'
]

def check_shared_dims(new_lp, lp, K_group):
    """
    Check that tensors with shared K_dims have the same plate dims
    """
    new_lp_dims = new_lp.names
    lp_dims = lp.names

    lp_K_dims = [dim for dim in lp_dims if is_K(dim)]

    new_lp_plate_dims = [dim for dim in new_lp_dims if is_plate(dim)]
    lp_plate_dims = [dim for dim in lp_dims if is_plate(dim)]

    if K_group in lp_K_dims:
        return new_lp_plate_dims == lp_plate_dims
    else:
        return True

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


class TraceSampleLogQ(Trace):
    """
    Samples a probabilistic program + evaluates log-probability.
    Usually used for sampling the approximate posterior.
    The latents may depend on the data (as in a VAE), but it doesn't make sense to "sample" data.
    Can high-level latents depend on plated lower-layer latents?  (I think so?)
    """

    def __init__(self, data=None, reparam=True, K_dim = None):
        super().__init__()
        if data is None:
            data = {}
        self.data = data
        self.sample = {}
        self.logp = {}
        self.reparam = reparam
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
        assert value.group == None
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
    def __init__(self, logq_trace, data=None, K_dim = None):
        self.sample = logq_trace.sample
        self.logq = logq_trace.logp
        self.data = data
        # maintains an ordered list of tensors as they are generated
        self.logp = {}
        self.dims = {}
        self.K_names2var = {}
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
        assert value.sample_K == True


        group = value.group
        K_name = 'K_{}'.format(group) if group is not None else f"K_{key}"
        #rename p and q (log q and q samples) here
        if key in self.sample:

            if has_dim(self.K_dim, get_dims(self.sample[key])):
                #If we have a plain K_dim, rename it to K_name
                if K_name not in [repr(dim) for dim in self.dims.values()]:
                    # If K_name dim already exists don't make another
                    dim = Dim(name=K_name, size=self.K_dim.size)
                else:
                    dim = self.dims[K_name]
                if key not in self.dims:
                    self.dims[key] = dim
                    self.dims[K_name] = dim
                self.sample[key] = self.sample[key].index(self.K_dim, self.dims[key])
                self.logq[key] = self.logq[key].rename(K=K_name)
                sample = self.sample[key]

            else:
                # if key not in self.dims:
                #     self.dims[key] = Dim(name=K_name, size=1)
                # self.sample[key] = self.sample[key].unsqueeze(0)[self.dims[key]]
                # logq_names = self.logq[key].names

                self.logq[key] = self.logq[key]#.rename(None).unsqueeze(0).refine_names(*((K_name,) + logq_names))
                sample = self.sample[key]

        if key not in self.K_names2var.get(K_name, []):
            self.K_names2var[K_name] = self.K_names2var.get(K_name, []) + [key]

        sample = self[key]
        new_lp = value.log_prob(sample)
        for log_p in self.logp.values():
            if group:
                assert check_shared_dims(new_lp, log_p, 'K_{}'.format(group))
        self.logp[key] = new_lp
