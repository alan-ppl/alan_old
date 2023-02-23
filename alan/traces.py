import math
from warnings import warn
import torch as t
import torch.distributions as td
from functorch.dim import dims, Dim

from .utils import *
from .timeseries import Timeseries
from .dist import Categorical, Uniform
from . import model

reserved_names = ("K", "N")
reserved_prefix = ("K_", "E_")
def check_not_reserved(x):
    reserved = (x in reserved_names) or (x[:2] in reserved_prefix)
    if reserved:
        raise Exception(f"You tried to use a variable or plate name '{x}'.  That name is reserved.  Specifically, we have reserved names {reserved_names} and reserved prefixes {reserved_prefix}.")

class AbstractTrace():
    def __init__(self, device):
        self.device = device

    def __call__(self, key, dist, plates=(), T=None, **kwargs):
        if isinstance(plates, str):
            plates = (plates,)

        self.sample_(key, dist, plates=plates, T=T, **kwargs)

    def __getitem__(self, key):
        return self.samples[key]

    def __contains__(self, key):
        in_data   = stack_key in self.data
        in_sample = stack_key in self.samples
        assert not (in_data and in_sample)
        return in_data or in_sample

    def filter_platedims(self, dims):
        platedims = set(self.platedims.values())
        return tuple(dim for dim in dims if (dim in platedims))

    def filter_Kdims(self, dims):
        self_Ks = self.Ks
        return tuple(dim for dim in dims if (dim in self_Ks))

    def extract_platedims(self, x):
        return self.filter_platedims(generic_dims(x))

    def extract_Kdims(self, x, exclude=None):
        result = self.filter_Kdims(generic_dims(x))
        result = set(result)
        if (exclude is not None) and (exclude in result):
            result.remove(exclude)
            result = tuple(result)
        return result

    def ones(self, *args, **kwargs):
        """
        Passes through to the underlying PyTorch method, but gets the right
        device
        """
        return t.ones(*args, **kwargs, device=self.device)

    def zeros(self, *args, **kwargs):
        """
        Passes through to the underlying PyTorch method, but gets the right
        device
        """
        return t.zeros(*args, **kwargs, device=self.device)

    @property
    def Ks(self):
        return set([*self.K_var.values(), *self.K_group.values()])


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
        #Dict mapping varname to K
        self.K_group = {}
        #Dict mapping varname to group
        self.group = {}

        self.group_parent_idxs = {}

    def sample_(self, key, dist, plates=(), T=None, group=None, multi_sample=True, sum_discrete=False):
        #Make sure the kwargs make sense
        if sum_discrete:
            raise Exception("We don't need an approximate posterior if sum_discrete=True")

        if multi_sample==False:
            warn("WARNING: multi_sample=False will break alot of things, including importance sampling, importance weighting, and RWS. Prefer grouped K's wherever possible. Though it is necessary to do Bayesian reasoning about parameters when we minibatch across latent variables")
            if group is not None:
                raise Exception(f"Doesn't make sense to group the variable {key} when multi_sample=False")

        if key in self.samples:
            raise Exception(f"Trying to sample {key}, but we have already have a variable with this name.")

        if (T is not None) and not isinstance(dist, Timeseries):
            raise Exception("T provided, but dist is not a Timeseries")
        if (T is None) and isinstance(dist, Timeseries):
            raise Exception("dist is a Timeseries, but T is not provided")
        if (T is not None) and (group is not None):
            raise Exception("Timeseries cannot currently be grouped")

        #If we've defined an approximate posterior for data then just skip it.
        #This is common if we're using P as Q
        if key in self.data:
            return None

        #Create new Kdim
        if (group is not None):
            self.group[key] = group
            #new group of K's
            if (group not in self.K_group):
                self.K_group[group] = Dim(f"K_{group}", K)
            Kdim = self.K_group[group]
            assert Kdim.size == self.trq.Kdim.size
        #non-grouped K's
        else:
            Kdim = Dim(f"K_{key}", self.K)
            self.K_var[key] = Kdim

        if T is not None:
            dist.set_trace_Tdim(self, self.platedims[T])

        sample_dims = platenames2platedims(self.platedims, plates)
        sample_dims = (Kdim, *sample_dims)
        sample = dist.sample(reparam=self.reparam, sample_dims=sample_dims, Kdim=Kdim)

        #Shouldn't matter if we run this on top of a Timeseries sample, as that sample should only
        #have one K (Kdim).
        sample = self.index_sample(sample, Kdim, group)

        logq = dist.log_prob(sample)
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

        #breakpoint()
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
        return f(x, self.platedims[plate])

    def mean(x, plate):
        return reduce_plate(t.mean, x, plate)


class TraceQCategorical(AbstractTraceQ):
    def parent_samples(self, plates, Kdim, K):
        return Categorical(t.ones(K.size)/K.size).sample(False, sample_dims=[Kdim, *plates])

    def logq(self, logq, Kdim):
        return log_meandims_exp(logq, self.extract_Kdims(logq, exclude=Kdim))

class TraceQPermutation(AbstractTraceQ):
    def parent_samples(self, plates, Kdim, K):
        assert Kdim.size == K.size
        return Uniform().sample(False, sample_dims=[Kdim, *plates]).argsort(Kdim)

    def logq(self, logq, Kdim):
        return log_meandims_exp(logq, self.extract_Kdims(logq, exclude=Kdim))

class TraceQSame(AbstractTraceQ):
    def parent_samples(self, plates, Kdim, K):
        idxs = t.arange(self.K)[Kdim].expand(*plates)
        return idxs

    def logq(self, logq, Kdim):
        plates = self.extract_platedims(logq)
        Ks = self.extract_Kdims(logq, exclude=Kdim)

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
    def __init__(self, platedims, N, device):
        super().__init__(device)
        self.Ns = () if (N is None) else (Dim('N', N),)
        self.platedims = platedims

        self.reparam = False

        self.samples = {}
        self.logp    = {}
        self.data    = {} #Unused, just here to make generic __contains__ and __getitem__ happy

    def sample_(self, key, dist, plates=(), T=None, group=None, sum_discrete=False):
        if T is not None:
            dist.set_trace_Tdim(self, self.platedims[T])

        sample_dims = [*self.Ns, *(self.platedims[plate] for plate in plates)]
        sample = dist.sample(reparam=self.reparam, sample_dims=sample_dims)
        self.samples[key] = sample
        self.logp[key] = dist.log_prob(sample)


#class TraceP(AbstractTraceP):
#    def __init__(self, trq):
#        super().__init__(trq.device)
#        self.trq = trq
#
#        self.samples = {}
#        self.logp = {}
#        self.logq = {}
#
#        self.groupname2dim = {}
#
#        #All Ks, including sum_discrete
#        self.Ks     = set()
#        #Only sum_discrete
#        self.Es     = set()
#
#    @property
#    def data(self):
#        return self.trq.data
#
#    @property
#    def platedims(self):
#        return self.trq.platedims
#
#    def sum_discrete(self, key, dist, plates):
#        """
#        Enumerates discrete variables.
#        """
#        if dist.dist_name not in ["Bernoulli", "Categorical"]:
#            raise Exception(f'Can only sum over discrete random variables with a Bernoulli or Categorical distribution.  Trying to sum over a "{dist.dist_name}" distribution.')
#
#        dim_plates    = set(dim for dim in dist.dims if (dim in self.platedims))
#        sample_plates = platenames2platedims(self.platedims, plates)
#        plates = list(dim_plates.union(sample_plates))
#
#        torch_dist = dist.dist(**dist.all_args)
#
#        values = torch_dist.enumerate_support(expand=False)
#        values = values.view(-1)
#        assert 1 == values.ndim
#        Edim = Dim(f'E_{key}', values.shape[0])
#        values = values[Edim]
#        #values is now just all a vector containing values in the support.
#
#        #Add a bunch of 1 dimensions.
#        idxs = (len(plates)*[None])
#        idxs.append(Ellipsis)
#        values = values[idxs]
#        #Expand them to full size
#        values = values.expand(*[plate.size for plate in plates])
#        #And name them
#        values = values[plates]
#        return values, Edim
#
#    def sample_(self, key, dist, group=None, plates=(), T=None, sum_discrete=False):
#        assert key not in self.logp
#        if T is not None:
#            dist.set_trace, Tdim(self, self.platedims[T])
#
#        if isinstance(dist, Timeseries) and (dist._inputs is not None):
#            warn("Generative models with timeseries with inputs are likely to be highly inefficient; if possible, try to rewrite the model so that timeseries doesn't have inputs.  For instance, you could marginalise the inputs and include them as noise in the transitions.")
#
#        if isinstance(dist, Timeseries) and (group is not None):
#            #Works around a limitation in importance sampling.
#            #Namely, we importance sample variables according to their order in P.
#            #If we have a plate and a timeseries which are grouped, then we have to sample the timeseries first
#            #that'll happen directly if the timeseries appears first in the probabilistic model,
#            #but we'll do something wrong if we try to sample the plate first.
#            if group in self.groupname2dim:
#                raise Exception(f"Timeseries '{key}' is grouped with another variable which is sampled first. Timeseries can be grouped, but must be sampled first")
#
#        if self.trq.contains_stack_key(key):
#            #We already have a value for the sample, either because it is
#            #in the data, or because we sampled the variable in Q.
#            if sum_discrete:
#                raise Exception("You have asked to sum over all settings of '{key}' (i.e. `sum_discrete=True`), but we already have a sample of '{key}' drawn from Q.  If you're summing over a discrete latent variable, you shouldn't provide a proposal / approximate posterior for that variable.")
#
#            sample = self.trq.get_stack_key(key)
#            logq = self.trq.logq[key] if key in self.trq.samples else None
#
#            #Check that plates match for sample from Q previous and P here
#            Q_sample_plates = set(self.filter_platedims(generic_dims(sample)))
#            P_sample_plates = set(self.filter_platedims(dist.dims)).union(platenames2platedims(self.platedims, plates))
#            if Q_sample_plates != P_sample_plates:
#                raise Exception(f"The plates for P and Q don't match for variable '{key}'.  Specifically, P has plates {tuple(P_sample_plates)}, while Q has plates {tuple(Q_sample_plates)}")
#
#        else:
#            assert sum_discrete
#            #Analytically sum out a discrete latent
#            if group is not None:
#                raise Exception("You have asked to sum over all settings of '{key}' (i.e. `sum_discrete=True`), but you have also provided a group.  This doesn't make sense.  Only variables sampled from Q can be grouped.")
#            sample, Kdim = self.sum_discrete(key, dist, plates)
#            logq = t.zeros_like(sample)
#            self.Ks.add(Kdim)
#            self.Es.add(Kdim)
#
#        dims_sample = set(generic_dims(sample))
#
#        minus_log_K = 0.
#
#        #If the sample has a trq.Kdim
#        has_Q_K = self.trq.Kdim in dims_sample
#        if has_Q_K:
#            #grouped K's
#            if (group is not None):
#                #new group of K's
#                if (group not in self.groupname2dim):
#                    self.groupname2dim[group] = Dim(f"K_{group}", self.trq.Kdim.size)
#                Kdim = self.groupname2dim[group]
#                assert Kdim.size == self.trq.Kdim.size
#            #non-grouped K's
#            else:
#                Kdim = Dim(f"K_{key}", self.trq.Kdim.size)
#
#            if Kdim not in self.Ks:
#                #New K-dimension.
#                minus_log_K = -math.log(self.trq.Kdim.size)
#                self.Ks.add(Kdim)
#
#            #Rename K -> K_groupname, or K_varname.
#            sample = sample.order(self.trq.Kdim)[Kdim]
#            logq = logq.order(self.trq.Kdim)[Kdim]
#
#        if key not in self.data:
#            self.samples[key] = sample
#        if logq is not None:
#            self.logq[key] = logq
#
#        #Timeseries needs to know the Kdim, but other distributions ignore it.
#        self.logp[key] = dist.log_prob_P(sample, Kdim=(Kdim if has_Q_K else None)) + minus_log_K
#
#class TracePGlobal(TraceP):
#    """
#    Incomplete method purely used for benchmarking.
#    e.g. doesn't do sampling from the prior.
#    """
#    def __init__(self, trq):
#        super().__init__(trq)
#        if isinstance(trq, TraceQTMC):
#            self.Ks = trq.Ks
#

class TraceP(AbstractTrace):
    def __init__(self, trq):
        super().__init__(trq.device)
        self.platedims = trq.platedims
        self.samples = trq.samples
        self.data = trq.data
        self.logq_group, self.logq_var = trq.finalize_logq()
        self.K_group = trq.K_group
        self.K_var = trq.K_var
        self.group = trq.group
        self.reparam = trq.reparam

        self.logp = {}

    def sample_(self, key, dist, group=None, plates=(), T=None, sum_discrete=False):
        if T is not None:
            dist.set_trace_Tdim(self, self.platedims[T])

        assert key not in self.logp
        assert (key in self.samples) != (key in self.data)

        sample = self.samples[key] if (key in self.samples) else self.data[key]
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
    def __init__(self, N, samples_train, data_train, data_all, covariates_train, covariates_all, platedims_train, platesizes_all, device):
        super().__init__(device)
        self.N = N

        self.samples_train = samples_train
        self.data_train = data_train
        covariates_train = covariates_train
        self.platedims_train = platedims_train

        #Either data_all or platesizes_all is not None, but not both.
        assert (data_all is None) != (platesizes_all is None)

        if platesizes_all is not None:
            self.platedims_all = extend_plates_with_sizes({}, platesizes_all)
            self.data_all      = {}
        if data_all is not None:
            self.platedims_all = extend_plates_with_named_tensors({}, data_all.values())
            self.data_all      = named2dim_tensordict(self.platedims_all, data_all)
        if covariates_all is not None:
            self.platedims_all = extend_plates_with_named_tensors(self.platedims_all, covariates_all.values())
            covariates_all      = named2dim_tensordict(self.platedims_all, covariates_all)
        else:
            assert False

        self.samples_all  = {}
        self.ll_all       = {}
        self.ll_train     = {}

        self.reparam      = False

        #Check that any new dimensions exist in training, and are bigger than those in training
        for platename, platedim_all in self.platedims_all.items():
            if platename not in self.platedims_train:
                raise Exception(f"Provided a plate dimension '{platename}' in platesizes_all or data_all which isn't present in the training data.")
            if platedim_all.size < self.platedims_train[platename].size:
                raise Exception(f"Provided a plate dimension '{platename}' in platesizes_all or data_all which is smaller than than the same plate in the training data (remember that the new data / plate sizes correspond to the training + test data)")

        #If any random variables from data_train are missing in data_all, fill them in
        for (rv, x) in self.data_train.items():
            if rv not in self.data_all:
                self.data_all[rv] = x

        #If any covariates from data_train are missing in data_all, fill them in
        for (rv, x) in covariates_train.items():
            if rv not in covariates_all:
                covariates_all[rv] = x

        #If any plates from platedims_train are missing in platedims_all, fill them in
        for (platename, platedim) in self.platedims_train.items():
            if platename not in self.platedims_all:
                self.platedims_all[platename] = platedim

        #Check that _something_ is bigger:
        data_bigger   = any(data_train[dataname].numel() < dat.numel() for (dataname, dat) in self.data_all.items())
        plates_bigger = any(platedims_train[platename].size < plate.size for (platename, plate) in self.platedims_all.items())

        if not (data_bigger or plates_bigger):
            raise Exception(f"None of the data tensors or plate sizes provided for prediction is bigger than those at training time.  Remember that the data/plate sizes are the sizes of train + 'test'")

        self.covariates_all =  covariates_all

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
        dimnames_all   = [name for (name, dim) in self.platedims_all.items()   if dim in dims_all]
        dimnames_train = [name for (name, dim) in self.platedims_train.items() if dim in dims_train]
        assert set(dimnames_all) == set(dimnames_train)
        dimnames = dimnames_all

        #Corresponding list of dims for all and train.
        dims_all   = [self.platedims_all[dimname]   for dimname in dimnames]
        dims_train = [self.platedims_train[dimname] for dimname in dimnames]
        dims_all.append(Ellipsis)
        dims_train.append(Ellipsis)
        return dims_all, dims_train

    def sample_(self, varname, dist, group=None, plates=(), T=None, sum_discrete=False):
        assert varname not in self.samples_all
        assert varname not in self.ll_all
        assert varname not in self.ll_train

        if T is not None:
            dist.set_trace_Tdim(self, self.platedims_all[T])

        if varname in self.data_all:
            #Compute predictive log-probabilities and put them in self.ll_all and self.ll_train
            self._sample_logp(varname, dist, plates)
        else:
            #Compute samples, and put them in self.sample_all
            self._sample_sample(varname, dist, plates)

    def _sample_sample(self, varname, dist, plates):
        sample_dims = platenames2platedims(self.platedims_all, plates)
        sample_dims.append(self.N)

        sample_all = dist.sample(reparam=self.reparam, sample_dims=sample_dims)
        sample_train = self.data_train[varname] if (varname in self.data_train) else self.samples_train[varname]

        dims_all, dims_train = self.corresponding_plates(sample_all, sample_train)
        sample_all   = generic_order(sample_all,   dims_all)   #Still torchdim, as it has N!
        sample_train = generic_order(sample_train, dims_train) #Still torchdim, as it has N!

        idxs = [slice(0, l) for l in sample_train.shape[:len(dims_all)]]
        idxs.append(Ellipsis)

        #Actually do the replacement in-place
        sample_all[idxs] = sample_train

        #Put torchdim back
        sample_all = sample_all[dims_all]

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

            sample_train, sample_test = split_train_test(sample_all, T_all, T_train, T_test)
            sample_init = sample_train.order(T_train)[-1]

            inputs = ()
            if dist._inputs is not None:
                inputs_train, inputs_test = split_train_test(dist._inputs, T_all, T_train, T_test)
                inputs = (inputs_test,)

            test_dist = Timeseries(sample_init, dist.transition, *inputs)
            test_dist.set_trace_Tdim(self, T_test)
            sample_test = test_dist.sample(reparam=self.reparam, sample_dims=sample_dims)
            sample_all = t.cat([sample_train.order(T_train), sample_test.order(T_test)], 0)[T_all]

        self.samples_all[varname] = sample_all

    def _sample_logp(self, varname, dist, plates):
        sample_all   = self.data_all[varname]
        sample_train = self.data_train[varname]

        dims_all, dims_train = self.corresponding_plates(sample_all, sample_train)

        sample_all_ordered   = generic_order(sample_all,   dims_all)
        sample_train_ordered = generic_order(sample_train, dims_train)

        idxs = [slice(0, l) for l in sample_train_ordered.shape[:len(dims_all)]]
        idxs.append(Ellipsis)

        # Check that data_all matches data_train
        #assert t.allclose(sample_all_ordered[idxs], sample_train_ordered)

        ll_all                 = dist.log_prob(sample_all)
        self.ll_all[varname]   = ll_all
        self.ll_train[varname] = generic_order(ll_all, dims_all)[idxs][dims_train]

def split_train_test(x, T_all, T_train, T_test):
    x_undim = x.order(T_all)
    return x_undim[:T_train.size][T_train], x_undim[T_train.size:][T_test]
