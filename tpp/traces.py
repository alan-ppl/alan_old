from warnings import warn
import torch as t
from functorch.dim import dims, Dim

from .utils import *
from .timeseries import Timeseries

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

    def sample(self, varname, dist, multi_samples=True, plate=None, T=None):
        assert varname not in self.samples

        if T is not None:
            dist.set_Tdim(self.plates[T])

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

    def sample(self, key, dist, multi_sample=True, plate=None, T=None):
        assert key not in self.data
        assert key not in self.samples
        assert key not in self.logq

        if multi_sample==False:
            warn("""

WARNING: multi_sample=False will break alot of things, 
including importance sampling, importance weighting, 
and RWS. Prefer grouped K's wherever possible.  
Though it is necessary to do Bayesian reasoning about 
parameters when we minibatch across latent variables

""")

        if T is not None:
            dist.set_Tdim(self.plates[T])

        sample_dims = []
        if plate is not None:
            sample_dims.append(self.plates[plate])
        if multi_sample:
            sample_dims.append(self.Kdim)

        sample = dist.sample(reparam=self.reparam, sample_dims=sample_dims)

        if not multi_sample:
            for d in generic_dims(sample):
                assert self.Kdim is not d, "Multiple samples are coming into this variable, so we can't stop it giving multiple samples at the output"

        self.samples[key] = sample
        self.logq[key] = dist.log_prob(sample)

    def reduce_plate(self, f, x, plate):
        """
        We may want an approximate posterior that samples the low-level latents
        plates before the high-level parameters.  We may also want the approximate
        posterior for the parameters to depend on e.g. the sampled values of the
        low-level latents.  As the latents will have a plate dimension that doesn't
        appear in the parameters, that means we'll need to reduce along a plate.
        But the user can't do that easily, because they don't have access to the 
        torchdim Dim for the plate, they only have a string.  The user therefore
        needs to use methods defined on this trace.

        Note that these do not exist on TraceP, because aggregating/mixing along
        a plate in the generative model will break things!
        """
        return f(x, self.plates[plate])

    def mean(x, plate):
        return reduce_plate(t.mean, x, plate)


class TraceP(AbstractTrace):
    def __init__(self, trq, memory_diagnostics=False):
        self.trq = trq
        self.memory_diagnostics=memory_diagnostics

        self.samples = {}
        self.logp = {}
        self.logq = {}

        self.groupname2dim = {}

        #Get plates from trq
        self.Ks     = set()

    @property
    def data(self):
        return self.trq.data

    def sample(self, key, dist, group=None, plate=None, T=None):
        assert key not in self.samples
        assert key not in self.logp

        if T is not None:
            dist.set_Tdim(self.trq.plates[T])

        dims_sample = set(generic_dims(self.trq[key]))
        
        #Check that the sample provided by TraceQ actually has the required plate.
        if plate is not None:
            assert self.trq.plates[plate] in dims_sample, "Plates in Q don't match those in P"

        has_k = self.trq.Kdim in dims_sample

        if isinstance(dist, Timeseries) and (dist._inputs is not None):
            warn("Generative models with timeseries with inputs are likely to be highly inefficient; if possible, try to eliminate the inputs (e.g. by marginalising them)")

        if isinstance(dist, Timeseries) and (group is not None):
            #Works around a limitation in importance sampling.
            #Namely, we importance sample variables according to their order in P.
            #If we have a plate and a timeseries which are grouped, then we have to sample the timeseries first
            #that'll happen directly if the timeseries appears first in the probabilistic model,
            #but we'll do something wrong if we try to sample the plate first.
            assert group not in self.groupname2dim, "Timeseries can be grouped, but must be the first thing sampled with a group"
            

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

            sample = self.trq[key]
            logq = self.trq.logq[key]

            # rename K_dim if it exists
            if has_k:
                sample = generic_order(sample, (self.trq.Kdim,))[Kdim]
                logq = generic_order(logq, (self.trq.Kdim,))[Kdim]

            self.samples[key] = sample
            self.logq[key] = logq

        #Timeseries needs to know the Kdim, but other distributions ignore it.
        self.logp[key] = dist.log_prob_P(sample, Kdim=(Kdim if has_k else None))


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

    def corresponding_plates(self, x_all, x_train):
        """
        x_all and x_train are tensors with plates, but the all and training plates
        have different sizes.  This returns two lists of torchdims, representing the
        plates for x_all and x_train.  It also checks that x_all and x_train have
        the same plates.
        """
        dims_all   = set(x_all.dims)   #Includes N!
        dims_train = set(x_train.dims) #Includes N!
        dimnames_all   = [name for (name, dim) in self.plates_all.items()   if dim in dims_all]
        dimnames_train = [name for (name, dim) in self.plates_train.items() if dim in dims_train]
        assert set(dimnames_all) == set(dimnames_train)
        dimnames = dimnames_all

        #Corresponding list of dims for all and train.
        dims_all   = [self.plates_all[dimname]   for dimname in dimnames]
        dims_train = [self.plates_train[dimname] for dimname in dimnames]
        dims_all.append(Ellipsis)
        dims_train.append(Ellipsis)
        return dims_all, dims_train

    def sample(self, varname, dist, multi_samples=True, plate=None):
        """
        We have three possible things: sample_tt, sample_at, sample_aa. 
        Here, t/a stand for train/all. The first t/a is for the previous
        plates.  The second t/a is for the current plate.  We treat them
        separately, because we need to be careful if plate is a timeseries

        Basic algorithm:
        draw sample_at from prior
        replace sample_tt in sample_at.
        sample extra to get sample_aa.
        """
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

        sample_train = self.data_train[varname] if (varname in self.data_train) else self.samples_train[varname]

        dims_all, dims_train = self.corresponding_plates(sample_all, sample_train)
        ##Get a unified list of dimension names.
        #dims_all   = set(sample_all.dims)   #Includes N!
        #dims_train = set(sample_train.dims) #Includes N!
        #dimnames_all   = [name for (name, dim) in self.plates_all.items()   if dim in dims_all]
        #dimnames_train = [name for (name, dim) in self.plates_train.items() if dim in dims_train]
        #assert set(dimnames_all) == set(dimnames_train)
        #dimnames = dimnames_all

        ##Corresponding list of dims for all and train.
        #dims_all   = [self.plates_all[dimname]   for dimname in dimnames]
        #dims_train = [self.plates_train[dimname] for dimname in dimnames]
        #dims_all.append(Ellipsis)
        #dims_train.append(Ellipsis)
        #Strip torchdim.
        sample_all   = generic_order(sample_all,   dims_all)   #Still torchdim, as it has N!
        sample_train = generic_order(sample_train, dims_train) #Still torchdim, as it has N!

        idxs = [slice(0, l) for l in sample_train.shape[:len(dims_all)]]
        idxs.append(Ellipsis)

        if varname in self.data_all:
            pass
            #assert t.allclose(sample_all[idxs], sample_train)
        else:
            #Actually do the replacement in-place
            sample_all[idxs] = sample_train

        #Put torchdim back
        sample_all = sample_all[dims_all]

        if varname in self.data_all:
            ll_all                 = dist.log_prob(sample_all)
            self.ll_all[varname]   = ll_all
            self.ll_train[varname] = generic_order(ll_all, dims_all)[idxs][dims_train]
        elif varname in self.data_train:
            self.data_all[varname] = sample_all
        else:
            self.samples_all[varname] = sample_all

    def _sample_logp(self, varname, dist, multi_samples=True, plate=None):
        sample_all = self.data_all[varname]

        ll_all                 = dist.log_prob(sample_all)
        self.ll_all[varname]   = ll_all
        self.ll_train[varname] = generic_order(ll_all, dims_all)[idxs][dims_train]
