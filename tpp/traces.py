import math
from warnings import warn
import torch as t
import torch.distributions as td
from functorch.dim import dims, Dim

from .utils import *
from .timeseries import Timeseries

reserved_names = ("K", "N")
reserved_prefix = ("K_", "E_")
def check_not_reserved(x):
    reserved = (x in reserved_names) or (x[:2] in reserved_prefix)
    if reserved:
        raise Exception(f"You tried to use a variable or plate name '{x}'.  That name is reserved.  Specifically, we have reserved names {reserved_names} and reserved prefixes {reserved_prefix}.")
    

class AbstractTrace():
    def __getitem__(self, key):
        assert key in self
        return self.samples[key] if (key in self.samples) else self.data[key]

    def __contains__(self, key):
        in_data   = key in self.data
        in_sample = key in self.samples
        assert not (in_data and in_sample)
        return in_data or in_sample

    def filter_platedims(self, dims):
        platedims = set(self.platedims.values())
        return [dim for dim in dims if dim in platedims]

    def check_varname(self, key):
        if key in self.samples:
            raise Exception(f"Trying to sample named {key}, but we have already sampled a variable with this name")
        check_not_reserved(key)

    def check_plate_present(self, key, plates, T):
        if isinstance(plates, str):
            plates = (plates,)
        for plate in plates:
            if (plate is not None) and (plate not in self.platedims):
                raise Exception(f"Plate '{plate}' on variable '{key}' not present.  Instead, we only have {tuple(self.platedims.keys())}.")
        if (T is not None) and (T not in self.platedims):
            raise Exception(f"Timeseries T '{T}' on variable '{key}' not present.  Instead, we only have {tuple(self.platedims.keys())}.")

    def check(self, key, plate, T):
        self.check_varname(key)
        self.check_plate_present(key, plate, T)

class TraceSample(AbstractTrace):
    """
    Draws samples from P.  Usually just used to sample fake data from the model.
    sizes is a dict mapping platenames to plate sizes

    If we want to draw multiple samples, we use samples.
    """
    def __init__(self, platesizes, N):
        super().__init__()
        self.platedims = extend_plates_with_sizes({}, platesizes)
        self.Ns = () if (N is None) else (Dim('N', N),)

        self.reparam = False

        self.data                 = {}
        self.samples              = {}

    def sample(self, key, dist, group=None, plates=(), T=None, sum_discrete=False):
        self.check(key, plates, T)

        if T is not None:
            dist.set_Tdim(self.platedims[T])

        sample_dims = [*self.Ns, *platenames2platedims(self.platedims, plates)]
        self.samples[key] = dist.sample(reparam=self.reparam, sample_dims=sample_dims)

def sample(P, platesizes=None, N=None, varnames=None):
    """Draw samples from a generative model (with no data).
    
    Args:
        P:        The generative model (a function taking a trace).
        plates:   A dict mapping platenames to integer sizes of that plate.
        N:        The number of samples to draw
        varnames: An iterable of the variables to return

    Returns:
        A dictionary mapping from variable name to sampled value, 
        represented as a named tensor (e.g. so that it is suitable 
        for use as data).
    """
    if platesizes is None:
        platesizes = {}
    tr = TraceSample(platesizes, N)
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
    def __init__(self, K, data, platedims, reparam):
        super().__init__()
        self.Kdim = Dim("K", K)

        self.data = data
        self.platedims = platedims

        self.reparam = reparam

        self.samples = {}
        self.logq = {}

    def sample(self, key, dist, multi_sample=True, plates=(), T=None):
        self.check(key, plates, T)
        if key in self.data:
            raise Exception(f"Q acts as a proposal / approximate posterior for latent variables, so we should only sample latent variables in Q.  However, we are sampling '{key}', which is data.")
        assert key not in self.logq

        if multi_sample==False:
            warn("WARNING: multi_sample=False will break alot of things, including importance sampling, importance weighting, and RWS. Prefer grouped K's wherever possible. Though it is necessary to do Bayesian reasoning about parameters when we minibatch across latent variables")

        if T is not None:
            dist.set_Tdim(self.platedims[T])

        Kdims = (self.Kdim,) if multi_sample else ()
        platedims = platenames2platedims(self.platedims, plates)
        sample_dims = (*Kdims, *platedims)

        sample = dist.sample(reparam=self.reparam, sample_dims=sample_dims)

        if not multi_sample:
            if any(self.Kdim is d for d in generic_dims(sample)):
                raise Exception(f"You have specified multi_samples=False on '{key}'. But we can't draw only a single sample for '{key}' as the specified approximate posterior depends on latent variables that themselves have multiple samples")

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
        return f(x, self.platedims[plate])

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

        #All Ks, including sum_discrete
        self.Ks     = set()
        #Only sum_discrete
        self.Es     = set()

    @property
    def data(self):
        return self.trq.data

    @property
    def platedims(self):
        return self.trq.platedims

    def sample_logQ_prior(self, dist, plates, Kdim):
        """
        When variables are omitted in TraceQ, we sample them from the prior.
        This only makes sense with multiple samples, which is nice as we no
        longer have the opportunity to set multi_sample=False in TraceQ.

        The basic strategy is to sample from the prior dist, then take the 
        "diagonal" for all K's.
        """

        #Don't depend on any enumerated variables (in which case sampling
        #from the prior doesn't make sense).
        assert all((dim not in self.Es) for dim in dist.dims)

        sample_dims = platenames2platedims(self.platedims, plates)

        all_Ks = set(self.Ks)
        if 0 == sum(dim in self.Ks for dim in dist.dims):
            #If there aren't any K's in the dist, then add a K.
            sample_dims.append(Kdim)
            all_Ks.add(Kdim)

        sample_all = dist.sample(self.trq.reparam, sample_dims)
        logq_all   = dist.log_prob(sample_all)

        Ks = [dim for dim in generic_dims(sample_all) if dim in all_Ks]
        idxs = len(Ks) * [range(Kdim.size)]

        sample = generic_order(sample_all, Ks)[idxs][Kdim]
        logq   = generic_order(logq_all,   Ks)[idxs][Kdim]
        return sample, logq

    def sum_discrete(self, key, dist, plates):
        """
        Enumerates discrete variables.
        """
        if dist.dist_name not in ["Bernoulli", "Categorical"]:
            raise Exception(f'Can only sum over discrete random variables with a Bernoulli or Categorical distribution.  Trying to sum over a "{dist.dist_name}" distribution.')

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
        values = values[plates]
        return values, Edim

    def sample(self, key, dist, group=None, plates=(), T=None, sum_discrete=False):
        self.check(key, plates, T)
        assert key not in self.logp
        if T is not None:
            dist.set_Tdim(self.platedims[T])

        if isinstance(dist, Timeseries) and (dist._inputs is not None):
            warn("Generative models with timeseries with inputs are likely to be highly inefficient; if possible, try to rewrite the model so that timeseries doesn't have inputs.  For instance, you could marginalise the inputs and include them as noise in the transitions.")

        if isinstance(dist, Timeseries) and (group is not None):
            #Works around a limitation in importance sampling.
            #Namely, we importance sample variables according to their order in P.
            #If we have a plate and a timeseries which are grouped, then we have to sample the timeseries first
            #that'll happen directly if the timeseries appears first in the probabilistic model,
            #but we'll do something wrong if we try to sample the plate first.
            if group in self.groupname2dim:
                raise Exception(f"Timeseries '{key}' is grouped with another variable which is sampled first. Timeseries can be grouped, but must be sampled first")

        if (key in self.trq):
            #We already have a value for the sample, either because it is 
            #in the data, or because we sampled the variable in Q.
            if sum_discrete:
                raise Exception("You have asked to sum over all settings of '{key}' (i.e. `sum_discrete=True`), but we already have a sample of '{key}' drawn from Q.  If you're summing over a discrete latent variable, you shouldn't provide a proposal / approximate posterior for that variable.")
             
            sample = self.trq[key]
            logq = self.trq.logq[key] if key in self.trq.samples else None

            #Check that plates match for sample from Q previous and P here
            Q_sample_plates = set(self.filter_platedims(generic_dims(sample)))
            P_sample_plates = set(self.filter_platedims(dist.dims)).union(platenames2platedims(self.platedims, plates))
            if Q_sample_plates != P_sample_plates:
                raise Exception(f"The plates for P and Q don't match for variable '{key}'.  Specifically, P has plates {tuple(P_sample_plates)}, while Q has plates {tuple(Q_sample_plates)}")

        else:
            #We don't already have a value for the sample, and we're either
            #going to sample from the prior, or enumerate a discrete variable
            if sum_discrete:
                #Analytically sum out a discrete latent
                if group is not None:
                    raise Exception("You have asked to sum over all settings of '{key}' (i.e. `sum_discrete=True`), but you have also provided a group.  This doesn't make sense.  Only variables sampled from Q can be grouped.")
                sample, Kdim = self.sum_discrete(key, dist, plates)
                logq = t.zeros_like(sample)
                self.Ks.add(Kdim)
                self.Es.add(Kdim)
            else:
                #Sample from prior
                sample, logq = self.sample_logQ_prior(dist, plates, self.trq.Kdim)
        
        dims_sample = set(generic_dims(sample))

        minus_log_K = 0.

        #If the sample has a trq.Kdim
        has_Q_K = self.trq.Kdim in dims_sample
        if has_Q_K:
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

            if Kdim not in self.Ks:
                #New K-dimension.
                minus_log_K = -math.log(self.trq.Kdim.size)
                self.Ks.add(Kdim)

            #Rename K -> K_groupname, or K_varname.
            sample = sample.order(self.trq.Kdim)[Kdim]
            logq = logq.order(self.trq.Kdim)[Kdim]

        if key not in self.data:
            self.samples[key] = sample
        if logq is not None:
            self.logq[key] = logq

        #Timeseries needs to know the Kdim, but other distributions ignore it.
        self.logp[key] = dist.log_prob_P(sample, Kdim=(Kdim if has_Q_K else None)) + minus_log_K


class TracePred(AbstractTrace):
    """
    Draws samples from P conditioned on samples from ...
    Usually just used to sample fake data from the model.

    post_rvs is posterior samples of all latents + training data.

    We can choose to provide data or sizes.
      If we provide data, then we compute test_ll
      If we provide sizes, then we compute predictive samples


    """
    def __init__(self, N, samples_train, data_train, data_all, platedims_train, platesizes_all):
        super().__init__()
        self.N = N

        self.samples_train = samples_train
        self.data_train = data_train
        self.platedims_train = platedims_train

        #Either data_all or platesizes_all is not None, but not both.
        assert (data_all is None) != (platesizes_all is None)
 
        if platesizes_all is not None:
            self.platedims_all = extend_plates_with_sizes({}, platesizes_all)
            self.data_all      = {}
        elif data_all is not None:
            self.platedims_all = extend_plates_with_named_tensors({}, data_all.values())
            self.data_all      = named2dim_tensordict(self.platedims_all, data_all)
        else:
            assert False

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

    def sample(self, varname, dist, group=None, plates=(), T=None, sum_discrete=False):
        assert varname not in self.samples_all
        assert varname not in self.ll_all
        assert varname not in self.ll_train

        if T is not None:
            dist.set_Tdim(self.platedims_all[T])

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
            test_dist.set_Tdim(T_test)
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
