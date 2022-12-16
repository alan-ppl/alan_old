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

        #Converts data to torchdim tensors, and adds plate dims to plates
        self.data, self.plates = named2dim_data(data, plates)


        #self.data = data
        #self.plates = plates
        self.reparam = reparam

        self.samples = {}
        self.logq = {}

    def sample(self, key, dist, multi_samples=True, plate=None):
        print(key)
        print(self.samples)
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

class TracePred(AbstractTrace):
    """
    Draws samples from P conditioned on samples from ...
    Usually just used to sample fake data from the model.

    post_rvs is posterior samples of all latents + training data.

    We can choose to provide data or sizes.
      If we provide data, then we compute test_ll
      If we provide sizes, then we compute predictive samples


    """
    def __init__(self, N, samples_train, data_train, plates_train, data_all=None, sizes_all=None):
        super().__init__()
        self.N = N

        self.samples_train = samples_train
        self.data_train = data_train
        self.plates_train = plates_train

        assert (data_all is None) != (sizes_all is None)
        if sizes_all is None:
            sizes_all = {}
        if data_all  is None:
            data_all  = {}

        self.plates_all = {}
        self.plates_all = insert_size_dict({}, sizes_all)
        self.plates_all = insert_named_tensors(self.plates_all, data_all.values())
        self.data_all   = named2dim_tensordict(self.plates_all, data_all)

        self.samples_all  = {}
        self.ll_all       = {}
        self.ll_train     = {}

        self.reparam      = False

    def __getitem__(self, key):
        in_data   = key in self.data_all
        in_sample = key in self.samples_all
        assert in_data or in_sample
        assert not (in_data and in_sample)
        return self.samples_all[key] if in_sample else self.data_all[key]

    def sample(self, varname, dist, multi_samples=True, plate=None):
        assert varname not in self.samples_all
        assert varname not in self.ll_all
        assert varname not in self.ll_train


        sample_dims = [self.N]
        if plate is not None:
            sample_dims.append(self.plates_all[plate])

        if varname in self.data_all:
            sample_all = self.data_all[varname]
        else:
            sample_all = dist.sample(reparam=self.reparam, sample_dims=sample_dims)
        sample_train = self.samples_train[varname]

        #Get a unified list of dimension names.
        dims_all   = set(sample_all.dims)   #Includes N!
        dims_train = set(sample_train.dims) #Includes N!
        dimnames_all   = [name for (name, dim) in self.plates_all.items()   if dim in dims_all]
        dimnames_train = [name for (name, dim) in self.plates_train.items() if dim in dims_train]
        assert set(dimnames_all) == set(dimnames_train)
        dimnames = dimnames_all

        #Corresponding list of dims for all and train.
        dims_all   = [self.plates_all[dimname]   for dimname in dimnames]
        dims_train = [self.plates_train[dimname] for dimname in dimnames]
        #Indices
        idxs = [slice(0, l) for l in sample_train.shape[:len(dimnames)]]
        idxs.append(Ellipsis)

        #Strip torchdim.
        sample_all   = generic_order(sample_all,   dims_all)   #Still torchdim, as it has N!
        sample_train = generic_order(sample_train, dims_train) #Still torchdim, as it has N!
        #Actually do the replacement in-place
        if varname == 'c':
            breakpoint()
        sample_all[idxs] = sample_train

        if varname in self.data_all:
            ll_all                 = dist.log_prob(sample_all)
            self.ll_all[varname]   = ll_all
            self.ll_train[varname] = generic_order(ll_all, dims_all)[idxs][dims_train]
        else:
            self.samples_all[varname] = sample_all
