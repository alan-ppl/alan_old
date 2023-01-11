import math
from warnings import warn
import torch as t
import torch.distributions as td
from functorch.dim import dims, Dim

from .utils import *
from .timeseries import Timeseries
from .dist import Categorical
from .tensors import NullTraceTensor, ValuedTraceTensor

reserved_names = ("K", "N")
reserved_prefix = ("K_", "E_")
def check_not_reserved(x):
    reserved = (x in reserved_names) or (x[:2] in reserved_prefix)
    if reserved:
        raise Exception(f"You tried to use a variable or plate name '{x}'.  That name is reserved.  Specifically, we have reserved names {reserved_names} and reserved prefixes {reserved_prefix}.")
    


class PQInputs():
    def __init__(self, model, args, kwargs):
        self.model = model
        self.args = args
        self.kwargs = kwargs

    def __call__(self, tr):
        self.model._forward(tr, *self.args, **self.kwargs)

class AbstractTrace():
    def __init__(self):
        self.stack = []

    def push_stack(self, name):
        self.stack.push(name)

    def pop_stack(self):
        self.stack.pop()

    def __getitem__(self, key):
        key = self.stack_key(key)
        assert key in self
        tensor = self.samples[key] if (key in self.samples) else self.data[key]
        return NullTraceTensor(key) if (tensor is None) else ValuedTraceTensor(tensor)

    def raw_getitem(self, key):
        assert key in self
        return self.samples[key] if (key in self.samples) else self.data[key]

    def __contains__(self, key):
        in_data   = key in self.data
        in_sample = key in self.samples
        assert not (in_data and in_sample)
        return in_data or in_sample

    def filter_platedims(self, dims):
        platedims = set(self.platedims.values())
        return [dim for dim in dims if (dim in platedims)]

    def check_plates(self, key, plates, T):
        if isinstance(plates, str):
            plates = (plates,)
        for plate in plates:
            if (plate is not None) and (plate not in self.platedims):
                raise Exception(f"Plate '{plate}' on variable '{key}' not present.  Instead, we only have {tuple(self.platedims.keys())}.")
        if (T is not None) and (T not in self.platedims):
            raise Exception(f"Timeseries T '{T}' on variable '{key}' not present.  Instead, we only have {tuple(self.platedims.keys())}.")

    def stack_key(self, key=None):
        result = '/'.join(self.stack)
        if key is not None:
            result = result + '/' + key
        return result

    @staticmethod
    def key_dist(key, dist):
        if dist is None:
            key, dist = dist, key
        return key, dist

    """
    Allows nested models and no key.
    """
    def P(self, key, dist=None, plates=(), T=None, sum_discrete=False):
        key, dist = self.key_dist(key, dist)

        if isinstance(dist, PQInputs):
            assert key is not None
            self.push_stack(key)
            assert self.stack_key() in self
            dist(self)
            self.pop_stack(key)
        else:
            stack_key = self.stack_key(key)
            self._P(stack_key, dist, plates=plates, T=T, sum_discrete=sum_discrete)

    def Q(self, key, dist=None, plates=(), T=None, multi_sample=True, group=None):
        key, dist = self.key_dist(key, dist)
        key = self.stack_key(key)

        assert not isinstance(dist, PQInputs)
        self._Q(key, dist, plates=plates, T=T, multi_sample=multi_sample, group=group)

    def PQ(self, key, dist=None, plates=(), T=None, multi_sample=True, group=None):
        """
        Used when the prior and approximate posterior are the same distribution.
        """
        key, dist = self.key_dist(key, dist)

        if isinstance(dist, PQInputs):
            assert key is not None
            self.push_stack(key)
            assert self.stack_key() not in self
            dist(self)
            assert self.stack_key() in self
            self.pop_stack(key)
        else:
            stack_key = self.stack_key(key)
            self._Q(stack_key, dist, plates=plates, T=T, multi_sample=multi_sample, group=group)
            self._P(stack_key, dist, plates=plates, T=T)

    def zeros(self, shape):
        return ValuedTraceTensor(t.zeros(shape))

    def ones(self, shape):
        return ValuedTraceTensor(t.ones(shape))

    #def P(self, key, dist, plates(), T=None, multi_sample=True, group=None):
    #    ...


class TraceSample(AbstractTrace):
    """
    Draws samples from P.  Usually just used to sample fake data from the model.
    sizes is a dict mapping platenames to plate sizes

    If we want to draw multiple samples, we use samples.
    """
    def __init__(self, platesizes, N):
        super().__init__()
        self.Ns = () if (N is None) else (Dim('N', N),)
        self.platedims = extend_plates_with_sizes({}, platesizes)

        self.reparam = False

        self.samples = {}
        self.logp    = {}
        self.data    = {} #Unused, just here to make generic __contains__ and __getitem__ happy

    def _P(self, key, dist, plates=(), T=None, sum_discrete=False):
        self.check_plates(key, plates, T)
        if dist.null:
            raise Exception(f"Trying to generate {key} in P, but it depends on a random variable that has only been sampled from Q")

        if T is not None:
            dist.set_Tdim(self.platedims[T])

        sample_dims = [*self.Ns, *platenames2platedims(self.platedims, plates)]
        sample = dist.sample(reparam=self.reparam, sample_dims=sample_dims)
        self.samples[key] = sample
        self.logp[key] = dist.log_prob(sample)

    def _Q(self, key, dist, plates=(), T=None, multi_sample=True, group=None):
        self.samples[key] = None

def sample_P(pq, platesizes=None, N=None, varnames=None):
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
    pq(tr)

    if varnames is None:
        varnames = tr.samples.keys()

    return {varname: dim2named_tensor(tr.samples[varname]) for varname in varnames}



class Trace(AbstractTrace):
    def __init__(self, K, data, platedims, reparam):
        super().__init__()
        self.Ksize = K
        self.data = data
        self.platedims = platedims
        self.reparam = reparam

        self.samples = {}
        self.logqs = {}
        self.logps = {}

        self.Ks     = set() #All Ks, including sum_discrete
        self.Es     = set() #Only sum_discrete
        self.groupname2dim = {}
        self.var2K  = {}

    def _Q(self, key, dist, plates=(), T=None, multi_sample=True, group=None):
        """
        Sets self.samples, self.logqs
        """
        self.check_plates(key, plates, T)
        check_not_reserved(key)

        assert (key in self.logqs) == (key in self.samples)

        if key in self.samples:
            raise Exception(f"Already have a Q for {key}")

        if key in self.data:
            raise Exception(f"Q acts as a proposal / approximate posterior for latent variables, so we should only sample latent variables in Q.  However, we are sampling '{key}', which is data.")

        if multi_sample==False:
            warn("WARNING: multi_sample=False will break alot of things, including importance sampling, importance weighting, and RWS. Prefer grouped K's wherever possible. Though it is necessary to do Bayesian reasoning about parameters when we minibatch across latent variables")

        if T is not None:
            dist.set_Tdim(self.platedims[T])

        if not multi_sample:
            if any(self.Kdim is d for d in generic_dims(sample)):
                raise Exception(f"You have specified multi_samples=False on '{key}'. But we can't draw only a single sample for '{key}' as the specified approximate posterior depends on latent variables that themselves have multiple samples")

        #Don't depend on any enumerated variables (in which case sampling
        #from the prior doesn't make sense).
        assert all((dim not in self.Es) for dim in dist.dims)

        plus_log_K = 0.
        sample_dims = platenames2platedims(self.platedims, plates)
        #If the sample has a trq.Kdim
        if multi_sample:
            #grouped K's
            if (group is not None):
                #new group of K's
                if (group not in self.groupname2dim):
                    self.groupname2dim[group] = Dim(f"K_{group}", self.trq.Kdim.size)
                Kdim = self.groupname2dim[group]
                assert Kdim.size == self.Ksize
            #non-grouped K's
            else:
                Kdim = Dim(f"K_{key}", self.Ksize)

            sample_dims = (Kdim, *sample_dims)

            if Kdim not in self.Ks:
                #New K-dimension.
                plus_log_K = math.log(self.Ksize)
                self.Ks.add(Kdim)

        self.var2K[key] = Kdim if multi_sample else None

        #diagonalise the arguments to the dist
        diag_args = {**dist.dim_args}
        for k, v in diag_args.items():
            Ks = [dim for dim in generic_dims(v) if dim in self.Ks]
            if 0 < len(Ks):
                idxs = len(Ks) * [range(Kdim.size)]
                diag_args[k] = generic_order(v, Ks)[idxs][Kdim]
        diag_args = {k: ValuedTraceTensor(v) for (k, v) in diag_args.items()}
        diag_dist = type(dist)(**diag_args)

        sample = diag_dist.sample(self.reparam, sample_dims)
        self.samples[key] = sample
        self.logqs[key] = diag_dist.log_prob(sample) + plus_log_K

    def sum_discrete(self, key, dist, plates, T):
        if dist.dist_name not in ["Bernoulli", "Categorical"]:
            raise Exception(f'Can only sum over discrete random variables with a Bernoulli or Categorical distribution.  Trying to sum over a "{dist.dist_name}" distribution.')

        dist_plates    = set(dim for dim in dist.dims if (dim in self.platedims))
        sample_plates = platenames2platedims(self.platedims, plates)
        plates = list(dist_plates.union(sample_plates))

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
        self.samples[key] = values
        self.Ks.add(Edim)
        self.Es.add(Edim)

    def _P(self, key, dist, plates=(), T=None, sum_discrete=False):
        self.check_plates(key, plates, T)
        if T is not None:
            dist.set_Tdim(self.platedims[T])

        if sum_discrete:
            self.sum_discrete(key, dist, plates, T)

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


        if not ((key in self.data) or (key in self.samples)):
             raise Exception(f"{key} must either be data, or have sum_discrete=True, or we have already sampled it in Q.")
        else:
             sample = self.raw_getitem(key)
             #Check that plates match for sample from Q previous and P here
             Q_sample_plates = set(self.filter_platedims(generic_dims(sample)))
             P_sample_plates = set(self.filter_platedims(dist.dims)).union(platenames2platedims(self.platedims, plates))
             if Q_sample_plates != P_sample_plates:
                 raise Exception(f"The plates for P and Q don't match for variable '{key}'.  Specifically, P has plates {tuple(P_sample_plates)}, while Q has plates {tuple(Q_sample_plates)}")

        #Think for sum_discrete!
        Kdim = self.var2K[key] if (key in self.samples) else None
        self.logps[key] = dist.log_prob_P(sample, Kdim=Kdim)



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
        #If any plates from platedims_train are missing in platedims_all, fill them in
        for (platename, platedim) in self.data_train.items():
            if platename not in self.platedims_all:
                self.platedims_all[platename] = platedim

        #Check that _something_ is bigger:
        data_bigger   = any(data_train[dataname].numel() < dat.numel() for (dataname, dat) in self.data_all.items())
        plates_bigger = any(platedims_train[platename].size < plate.size for (platename, plate) in self.platedims_all.items())

        if not (data_bigger or plates_bigger):
            raise Exception(f"None of the data tensors or plate sizes provided for prediction is bigger than those at training time.  Remember that the data/plate sizes are the sizes of train + 'test'")
            

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

    def sample(self, varname, dist, group=None, plates=(), T=None, sum_discrete=False, delayed_Q=None):
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


#class TraceQTMC(AbstractTrace):
#    def __init__(self, K, data, platedims, reparam):
#        super().__init__()
#        self.K = K
#
#        self.data = data
#        self.platedims = platedims
#
#        self.reparam = reparam
#
#        self.samples = {}
#        self.logq = {}
#
#        self.groupname2dim = {}
#        self.Ks     = set()
#
#
#    def sample(self, key, dist, group=None, plates=(), T=None):
#        if T is not None:
#            dist.set_Tdim(self.platedims[T])
#
#        #grouped K's
#        if (group is not None):
#            #new group of K's
#            if (group not in self.groupname2dim):
#                self.groupname2dim[group] = Dim(f"K_{group}", self.trq.Kdim.size)
#            Kdim = self.groupname2dim[group]
#            assert Kdim.size == self.trq.Kdim.size
#        #non-grouped K's
#        else:
#            Kdim = Dim(f"K_{key}", self.K)
#
#        plus_log_K = 0.
#        if Kdim not in self.Ks:
#            #New K-dimension.
#            plus_log_K = math.log(self.K)
#            self.Ks.add(Kdim)
#
#        sample_dims = platenames2platedims(self.platedims, plates)
#        nks = sum((dim in self.Ks) for dim in dist.dims)
#        if nks == 0:
#            sample_dims = (Kdim, *sample_dims)
#        sample = dist.sample(reparam=self.reparam, sample_dims=sample_dims)
#        logq = dist.log_prob(sample)
#
#        if 0 < nks:
#            plates = self.filter_platedims(sample.dims)
#            Ks = [dim for dim in sample.dims if (dim in self.Ks)]
#            idxs = [Categorical(t.ones(self.K)/self.K).sample(False, sample_dims=[Kdim, *plates]) for K in Ks]
#            sample = sample.order(*Ks)[idxs]
#            logq   = mean_dims(dist.log_prob(sample).exp(), Ks).log()
#
#        self.samples[key] = sample
#        self.logq[key] = logq + plus_log_K
