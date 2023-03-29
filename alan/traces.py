import math
from warnings import warn
import torch as t
import torch.distributions as td
from functorch.dim import dims, Dim

from .utils import *
from .timeseries import Timeseries
from .dist import Categorical, Uniform
from . import model

class GetItem():
    def __init__(self, data, samples, platedims):
        self.data = data
        self.samples = samples
        self.platedims = platedims

    def __getitem__(self, key):
        if not key in self:
            raise Exception(f"Called tr['{key}'], but {key} not present in data or samples")
        if key in self.data:
            return self.data[key]
        elif key in self.samples:
            return self.samples[key]
        else:
            assert False

    def __contains__(self, key):
        in_data   = key in self.data
        in_sample = key in self.samples
        result = in_data + in_sample
        assert result != 2
        return result == 1

class AbstractTrace(GetItem):
    def __init__(self, device):
        self.device = device

    def __call__(self, key, dist, plates=(), T=None, **kwargs):
        if isinstance(plates, str):
            plates = (plates,)

        for plate in plates:
            if plate not in self.platedims:
                raise Exception(
                    f"Trying to sample '{key}' with plate '{plate}', but"
                    f"size of plate {plate} is unknown"
                )

        if key in self.samples:
            raise Exception("Trying to sample '{key}', but '{key}' has already been sampled")

        if (T is not None) and not isinstance(dist, Timeseries):
            raise Exception("T provided, but dist is not a Timeseries")
        if (T is None) and isinstance(dist, Timeseries):
            raise Exception("dist is a Timeseries, but T is not provided")

        self.sample_(key, dist, plates=plates, T=T, **kwargs)

    def filter_platedims(self, dims):
        platedims = set(self.platedims.values())
        return tuple(dim for dim in dims if (dim in platedims))

    def filter_Kdims(self, dims):
        self_Ks = self.Ks
        return tuple(dim for dim in dims if (dim in self_Ks))

    def extract_platedims(self, x):
        return self.filter_platedims(generic_dims(x))

    def extract_Kdims(self, x, exclude=None, extra_K=None):
        result = self.filter_Kdims(generic_dims(x))
        result = set(result)
        if extra_K is not None:
            assert extra_K is not exclude
            result.add(extra_K)
        if (exclude is not None) and (exclude in result):
            result.remove(exclude)
            result = tuple(result)
        return result

    def ones(self, *args, **kwargs):
        """
        Passes through to the underlying PyTorch method, but gets the right device
        """
        return t.ones(*args, **kwargs, device=self.device)

    def zeros(self, *args, **kwargs):
        """
        Passes through to the underlying PyTorch method, but gets the right device
        """
        return t.zeros(*args, **kwargs, device=self.device)

    @property
    def Ks(self):
        return set([*self.K_var.values(), *self.K_group.values()])

    def key2Kdim(self, key):
        if key in self.K_var:
            return self.K_var[key]
        if key in self.group:
            return self.K_group[self.group[key]]
        else:
            return None


class AbstractTraceQ(AbstractTrace):
    def __init__(self, K, data, platedims, reparam, device):
        super().__init__(device)
        self.K = K

        self.data = data
        self.platedims = platedims

        self.reparam = reparam

        self.samples = {}
        self.logq_var = {}
        self.logq_group = {}

        #Dict mapping varname to K
        self.K_var = {}
        #Dict mapping groupname to K
        self.K_group = {}
        #Dict mapping varname to groupname
        self.group = {}
        #Group -> parent_varname -> idxs
        self.group_parent_idxs = {}

    def sample_(self, key, dist, plates=(), T=None, group=None, multi_sample=True, sum_discrete=False):
        #Make sure the kwargs make sense
        if sum_discrete:
            raise Exception("We don't need an approximate posterior if sum_discrete=True")

        #Sometimes we want to use the prior as an approximate posterior.
        #The prior will specify how to sample the data.  But we don't
        #need to sample the data under the approximate posterior, so we
        #just ignore the call.
        if key in self.data:
            return None

        if multi_sample==False:
            warn(
                "WARNING: multi_sample=False will break alot of things, "
                "including importance sampling, importance weighting, "
                "and RWS. Prefer grouped K's wherever possible. Though "
                "it is necessary to do Bayesian reasoning about parameters "
                "when we minibatch across latent variables"
            )
        if (multi_sample==False) and (group is not None):
            raise Exception(f"Doesn't make sense to group the variable {key} when multi_sample=False")

        if (T is not None) and (group is not None):
            raise Exception("Timeseries cannot currently be grouped")

        #Create new Kdim
        if (group is not None):
            self.group[key] = group
            #new group of K's
            if (group not in self.K_group):
                self.K_group[group] = Dim(f"Kgroup_{group}", self.K)
            Kdim = self.K_group[group]
        #non-grouped K's
        else:
            Kdim = Dim(f"Kvar_{key}", self.K)
            self.K_var[key] = Kdim

        if T is not None:
            dist.set_trace_Tdim(self, self.platedims[T])

        sample_dims = platenames2platedims(self.platedims, plates)
        sample_dims = (Kdim, *sample_dims)

        #Draw one sample for each K in the parameters of dist,
        sample = dist.sample(reparam=self.reparam, sample_dims=sample_dims, Kdim=Kdim)
        #However, the resulting sample should have only one K,
        #corresponding to this variable's Kvar or Kgroup, and
        #we get that by selecting a value of each of the
        #previous Ks
        sample = self.index_sample(sample, Kdim, group)

        logq = dist.log_prob(sample, Kdim=Kdim)
        self.samples[key] = sample

        if group is not None:
            self.logq_group[group] = self.logq_group.get(key, 0.) + logq
        else:
            self.logq_var[key] = logq

    def index_sample(self, sample, Kdim, group):
        plates = self.extract_platedims(sample)
        Ks = self.extract_Kdims(sample, exclude=Kdim)

        if group is not None:
            idxs = self.group_parent_idxs[group]
        else:
            idxs = {}

        for K in Ks:
            if K not in idxs:
                idxs[K] = self.parent_samples(plates, Kdim, K)

        if 0 < len(Ks):
            sample = sample.order(*Ks)[[idxs[K] for K in Ks]]
        return sample

    def finalize_logq(self):
        """
        Need to post-process logq's for the mixture over parent particles.
        But we can't do this as we go along, because we need to combine all
        log-probability tensors within a group first
        """
        logq_group = {}
        for (k, v) in self.logq_group.items():
            logq_group[k] = self.logq(v, self.K_group[k])

        logq_var = {}
        for (k, v) in self.logq_var.items():
            logq_var[k] = self.logq(v, self.K_var[k])
        return logq_group, logq_var

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
        assert isinstance(plate, str)
        return f(x, self.platedims[plate])

    def mean(x, plate):
        return reduce_plate(t.mean, x, plate)


class TraceQCategorical(AbstractTraceQ):
    def parent_samples(self, plates, Kdim, K):
        return Categorical(t.ones(K.size)/K.size).sample(False, sample_dims=[Kdim, *plates])

    def logq(self, logq, Kdim, extra_K=None):
        return logmeanexp_dims(logq, self.extract_Kdims(logq, exclude=Kdim, extra_K=extra_K))

class TraceQPermutation(AbstractTraceQ):
    def parent_samples(self, plates, Kdim, K):
        assert Kdim.size == K.size
        return Uniform(0,1).sample(False, sample_dims=[Kdim, *plates]).argsort(Kdim)

    def logq(self, logq, Kdim, extra_K=None):
        return logmeanexp_dims(logq, self.extract_Kdims(logq, exclude=Kdim, extra_K=extra_K))

class TraceQSame(AbstractTraceQ):
    def parent_samples(self, plates, Kdim, K):
        idxs = t.arange(self.K)[Kdim].expand(*plates)
        return idxs

    def logq(self, logq, Kdim, extra_K=None):
        plates = self.extract_platedims(logq)
        Ks = self.extract_Kdims(logq, exclude=Kdim, extra_K=extra_K)

        if len(Ks) > 0:
            idxs = [self.parent_samples(plates, Kdim, K) for K in Ks]
            logq = logq.order(*Ks)[idxs]
        return logq


class TraceSample(AbstractTrace):
    """
    Draws samples from P.  Usually just used to sample fake data from the model.
    sizes is a dict mapping platenames to plate sizes

    If we want to draw multiple samples, we use samples.
    """
    def __init__(self, N, platedims, device):
        super().__init__(device)
        self.Ns = () if (N is None) else (Dim('N', N),)
        self.platedims = platedims

        self.reparam = False

        self.samples = {}
        self.logp    = {}
        self.data    = {} #Unused, just here to make generic __contains__ and __getitem__ happy

    def sample_(self, key, dist, plates=(), T=None, group=None, sum_discrete=False):
        del group, sum_discrete

        if T is not None:
            dist.set_trace_Tdim(self, self.platedims[T])

        sample_dims = [*self.Ns, *(self.platedims[plate] for plate in plates)]
        sample = dist.sample(reparam=self.reparam, sample_dims=sample_dims)
        self.samples[key] = sample

    @property
    def Ks(self):
        return set()

class TraceP(AbstractTrace):
    def __init__(self, trq):
        super().__init__(trq.device)
        self.platedims = trq.platedims
        self.data = trq.data
        self.samples_q = trq.samples
        self.logq_group, self.logq_var = trq.finalize_logq()
        self.K_group = trq.K_group
        self.K_var = trq.K_var
        self.group = trq.group
        self.reparam = trq.reparam

        self.samples = {}
        self.logp = {}
        self.Es = set()
        self.sum_discrete_varnames = set()

        #set of timeseries dimensions (as strings)
        self.Tdim2Ks = {}
        self.Tvar2Tdim = {}
        self.used_platenames = set()

    def sum_discrete(self, key, dist, plates):
        """
        Enumerates discrete variables.
        """
        if dist.dist not in [t.distributions.Bernoulli, t.distributions.Categorical]:
            raise Exception(
                f'Can only sum over discrete random variables with a '
                f'Bernoulli or Categorical distribution.  Trying to ' 
                f'sum over a "{dist}" distribution.'
            )

        dim_plates    = set(dim for dim in dist.dims if (dim in self.platedims))
        sample_plates = platenames2platedims(self.platedims, plates)
        plates = list(dim_plates.union(sample_plates))

        torch_dist = dist.dist(**dist.all_args)

        values = torch_dist.enumerate_support(expand=False)
        values = values.view(-1)
        assert 1 == values.ndim
        Edim = Dim(f'E_{key}', values.shape[0])
        values = values[Edim]
        #values is now just all a vector containing values in the support.

        #Add a bunch of 1 dimensions.
        idxs = (len(plates)*[None])
        idxs.append(Ellipsis)
        values = values[idxs]
        #Expand them to full size
        values = values.expand(*[plate.size for plate in plates])
        #And name them
        values = generic_getitem(values, plates)
        return values, Edim

    def sample_(self, key, dist, group=None, plates=(), T=None, sum_discrete=False):
        if sum_discrete and (key in self):
            raise Exception(
                f"Trying to sum over {key}, but variable already present in"
                f"either data or samples"
            )

        if (key not in self.samples_q) and (key not in self.data) and (not sum_discrete):
            raise Exception(
                f"Trying to compute log-prob for '{key}' in the generative model (P), "
                f"but '{key}' is not in data, and was not sampled in Q"
            )

        self.used_platenames = self.used_platenames.intersection(plates)

        if T is not None:
            if T in self.used_platenames:
                raise Exception(
                    "Timeseries must be the first thing sampled with the T-dimension; "
                    "you can sample plates later with T, but not earlier.  This is to "
                    "ensure that importance sampling works. "
                )
            Tdim = self.platedims[T]
            dist.set_trace_Tdim(self, Tdim)

            self.Tdim2Ks[Tdim] = (self.key2Kdim(dist.initial_state_key), self.key2Kdim(key))
            self.Tvar2Tdim[key] = Tdim

        if sum_discrete:
            sample, Edim = self.sum_discrete(key, dist, plates)
            self.Es.add(Edim)
            self.logp[key] = dist.log_prob(sample, Kdim=Edim)
            self.samples[key] = sample
            self.sum_discrete_varnames.add(key)
        else:
            if key in self.samples_q:
                self.samples[key] = self.samples_q[key]
            sample = self.samples_q[key] if (key in self.samples) else self.data[key]
            self.logp[key] = dist.log_prob(sample, Kdim=self.key2Kdim(key))



class TracePred(AbstractTrace):
    """
    Draws samples from P conditioned on samples from ...
    Usually just used to sample fake data from the model.
    post_rvs is posterior samples of all latents + training data.
    We can choose to provide data or sizes.
      If we provide data, then we compute test_ll
      If we provide sizes, then we compute predictive samples
    """
    def __init__(self, N, samples_train, data_train, data_all, platedims_train, platedims_all, device):
        super().__init__(device)
        self.N = N

        self.train = GetItem(data_train, samples_train, platedims_train)
        self.platedims_train = platedims_train

        self.samples = {}
        self.data = data_all
        self.platedims = platedims_all

        self.samples  = {}
        self.ll_all       = {}
        self.ll_train     = {}

        self.reparam      = False

        #Check that any new dimensions exist in training, and are bigger than those in training
        for platename, platedim_all in self.platedims.items():
            if platename not in self.train.platedims:
                raise Exception(f"Provided a plate dimension '{platename}' in platesizes_all or data which isn't present in the training data.")
            if platedim_all.size < self.train.platedims[platename].size:
                raise Exception(f"Provided a plate dimension '{platename}' in platesizes_all or data which is smaller than than the same plate in the training data (remember that the new data / plate sizes correspond to the training + test data)")

        #If any plates from platedims_train are missing in platedims, fill them in
        for (platename, platedim) in self.train.platedims.items():
            if platename not in self.platedims:
                self.platedims[platename] = platedim

        plates_bigger = any(platedims_train[platename].size < plate.size for (platename, plate) in self.platedims.items())
        if not plates_bigger:
            raise Exception(f"None of the data tensors or plate sizes provided for prediction is bigger than those at training time.  Remember that the data/plate sizes are the sizes of train + 'test'")

    def corresponding_plates(self, x_all, x_train):
        """
        x_all and x_train are tensors with plates, but the all and training plates
        have different sizes.  This returns two lists of torchdims, representing the
        plates for x_all and x_train.  It also checks that x_all and x_train have
        the same plates.
        """
        dims_all   = set(x_all.dims)   #Includes N!
        dims_train = set(x_train.dims) #Includes N!
        dimnames_all   = [name for (name, dim) in self.platedims.items()   if dim in dims_all]
        dimnames_train = [name for (name, dim) in self.train.platedims.items() if dim in dims_train]
        assert set(dimnames_all) == set(dimnames_train)
        dimnames = dimnames_all

        #Corresponding list of dims for all and train.
        dims_all   = [self.platedims[dimname]   for dimname in dimnames]
        dims_train = [self.train.platedims[dimname] for dimname in dimnames]
        return dims_all, dims_train

    def sample_(self, varname, dist, group=None, plates=(), T=None, sum_discrete=False):
        assert varname not in self.samples
        assert varname not in self.ll_all
        assert varname not in self.ll_train

        if T is not None:
            dist.set_trace_Tdim(self, self.platedims[T])

        if varname in self.data:
            #Compute predictive log-probabilities and put them in self.ll_all and self.ll_train
            self._sample_logp(varname, dist, plates)
        else:
            #Compute samples, and put them in self.sample
            self._sample_sample(varname, dist, plates)

    def _sample_sample(self, varname, dist, plates):
        sample_dims = platenames2platedims(self.platedims, plates)
        sample_dims.append(self.N)
        sample = dist.sample(reparam=self.reparam, sample_dims=sample_dims)
        sample_train = self.train[varname]

        dims_all, dims_train = self.corresponding_plates(sample, sample_train)
        sample       = generic_order(sample,       dims_all)   #Still torchdim, as it has N!
        sample_train = generic_order(sample_train, dims_train) #Still torchdim, as it has N!

        #idxs = [slice(0, l) for l in sample_train.shape[:len(dims_all)]]
        idxs = [slice(0, dim.size) for dim in dims_train]

        #Actually do the replacement in-place
        #sample[idxs] = sample_train
        generic_setitem(sample, idxs, sample_train)

        #Put torchdim back
        #sample = sample[dims_all]
        sample = generic_getitem(sample, dims_all)

        if isinstance(dist, Timeseries):
            #Throw away the "test" part of the timeseries, and resample. Note that
            #we sampled all (including test) of the timeseries in the first place
            #because any inputs would be all (including test).  We could modify the
            #inputs earlier on, but that would involve timeseries-specific branching
            #in two places.
            T_all = dist.Tdim
            T_idx = next(i for (i, dim) in enumerate(dims_all) if dim is T_all)
            T_train = dims_train[T_idx]
            T_test = Dim('T_test', T_all.size - T_train.size)

            sample_train, _ = split_train_test(sample, T_all, T_train, T_test)
            sample_init = sample_train.order(T_train)[-1]

            inputs_test = tuple(split_train_test(x, T_all, T_train, T_test)[1] for x in dist._inputs)
            test_dist = Timeseries.pred_init(sample_init, dist.transition, T_test, inputs_test)
            sample_test = test_dist.sample(reparam=self.reparam, sample_dims=sample_dims)
            sample = t.cat([sample_train.order(T_train), sample_test.order(T_test)], 0)[T_all]


        self.samples[varname] = sample

    def _sample_logp(self, varname, dist, plates):
        sample = self.data[varname]
        sample_train = self.train.data[varname]

        dims_all, dims_train = self.corresponding_plates(sample, sample_train)

        sample_ordered       = generic_order(sample,       dims_all)
        sample_train_ordered = generic_order(sample_train, dims_train)

        idxs = [slice(0, dim.size) for dim in dims_train]

        ll_all                 = dist.log_prob(sample)
        self.ll_all[varname]   = ll_all

        ll_train = generic_order(ll_all, dims_all)
        ll_train = generic_getitem(ll_train, idxs)
        ll_train = generic_getitem(ll_train, dims_train)
        self.ll_train[varname] = ll_train

def split_train_test(x, T_all, T_train, T_test):
    x_undim = x.order(T_all)
    return x_undim[:T_train.size][T_train], x_undim[T_train.size:][T_test]
