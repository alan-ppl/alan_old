import torch as t
from functorch.dim import dims, Dim

from .utils import *

class AbstractTrace():
    def __getitem__(self, key):
        in_data   = key in self.data
        in_sample = key in self.samples
        assert in_data or in_sample
        assert not (in_data and in_sample)
        return self.samples[key] if in_sample else self.data[key]

class TraceSample(AbstractTrace):
    """
    Draws samples from P.  Usually just used to sample fake data from the model.
    sizes is a dict mapping platenames to plate sizes
    """
    def __init__(self, sizes=None):
        super().__init__()
        self.plates = insert_size_dict({}, sizes)
        self.reparam = False

        self.data                 = {} 
        self.samples              = {}

    def sample(self, varname, dist, multi_samples=True, plate=None):
        assert varname not in self.samples
            
        sample_dims = [] if plate is None else [self.plates[plate]]
        self.samples[varname] = dist.sample(reparam=self.reparam, sample_dims=sample_dims)

def sample(P, sizes=None, varnames=None):
    if sizes is None:
        sizes = {}
    tr = TraceSample(sizes)
    P(tr)

    if varnames is None:
        varnames = tr.samples.keys()

    return {varname: dim2named_tensor(tr.samples[varname]) for varname in varnames}


class TraceQ(AbstractTrace):
    """
    Samples a probabilistic program + evaluates log-probability.
    Usually used for sampling the approximate posterior.
    The latents may depend on the data (as in a VAE), but it doesn't make sense to "sample" data.
    Can high-level latents depend on plated lower-layer latents?  (I think so?)
    """
    def __init__(self, K, data, plates, reparam):
        super().__init__()
        self.Kdim = Dim("K", K)
        self.data = data
        self.plates = plates
        self.reparam = reparam

        self.samples = {}
        self.logq = {}

    def sample(self, key, dist, multi_samples=True, plate=None):
        assert key not in self.data
        assert key not in self.samples
        assert key not in self.logq
            
        sample_dims = []
        if plate is not None:
            sample_dims.append(self.plates[plate])
        if multi_samples:
            sample_dims.append(self.Kdim)

        sample = dist.sample(reparam=self.reparam, sample_dims=sample_dims)
        if not multi_samples:
            assert self.Kdim not in sample.dims, "Multiple samples are coming into this variable, so we can't stop it giving multiple samples at the output"

        self.samples[key] = sample
        self.logq[key] = dist.log_prob(sample)
        

class TraceP(AbstractTrace):
    def __init__(self, trq):
        self.trq = trq

        self.samples = {}
        self.logp = {}
        self.logq = {}

        self.groupname2dim = {}

        #Get plates from trq
        self.Ks     = set()

    @property
    def data(self):
        return self.trq.data

    def sample(self, key, dist, group=None, plate=None):
        assert key not in self.samples
        assert key not in self.logp

        #data
        if key in self.data: 
            sample = self.data[key]
        #latent variable
        else: 
            #grouped K's
            if (group is not None): 
                #new group of K's
                if (group not in self.groupname2dim):
                    self.groupname2dim[group] = Dim(f"K_{group}", self.trq.Kdim.size)
                Kdim = self.groupname2dim[group]
                assert Kdim.size == self.trq.Kdim.size
            #non-grouped K's
            else:
                Kdim = Dim(f"K_{key}", self.trq.Kdim.size)
            self.Ks.add(Kdim)

            sample_q = self.trq[key]
            sample = sample_q.order(self.trq.Kdim)[Kdim]
            self.samples[key] = sample

            self.logq[key] = self.trq.logq[key].order(self.trq.Kdim)[Kdim]

        self.logp[key] = dist.log_prob(sample)
