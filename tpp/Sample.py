from .utils import *
from .dist import Categorical
from .timeseries import TimeseriesLogP, flatten_tslp_list


class Sample():
    """
    Does error checking on the log-ps, and does the tensor product.

    TODO:
      Check that latents (in samples) appear in logps and logqs
      Check that data appears in logps but not logqs
      Check that all dims are something (plate, timeseries, K)
    """
    def __init__(self, trp):
        self.trp = trp

        for lp in [*trp.logp.values(), *trp.logq.values()]:
            if isinstance(lp, TimeseriesLogP):
                assert lp.first.shape == () and lp.rest.shape == ()
            else:
                assert lp.shape == ()


        for (rv, lq) in trp.logq.items():
            #check that any rv in logqs is also in logps
            assert rv in trp.logp

            lp = trp.logp[rv]
            lq = trp.logq[rv]

            # check same plates/timeseries appear in lp and lq
            lp_notK = [dim for dim in lp.dims if not self.is_K(dim)]
            lq_notK = [dim for dim in lq.dims if not self.is_K(dim)]
            assert set(lp_notK) == set(lq_notK)


        #Assumes that self.lps come in ordered
        self.plates = set(trp.trq.plates.values())
        self.ordered_plate_dims = [dim for dim in unify_dims(trp.logp.values()) if self.is_plate(dim)]
        self.ordered_plate_dims = [None, *self.ordered_plate_dims]

    def is_plate(self, dim):
        return dim in self.plates

    def is_K(self, dim):
        return dim in self.trp.Ks

    def tensor_product(self, detach_p=False, detach_q=False, extra_log_factors=()):
        """
        Sums over plates, starting at the lowest plate.
        The key exported method.
        """
        logps = self.trp.logp
        logqs = self.trp.logq

        if detach_p:
            logps = {n:lp.detach() for (n,lp) in logps.items()}
        if detach_q:
            logqs = {n:lq.detach() for (n,lq) in logqs.items()}

        tensors = [*logps.values(), *[-lq for lq in logqs.values()], *extra_log_factors]

        ## Convert tensors to Float64
        tensors = [x.to(dtype=t.float64) for x in tensors]

        #iterate from lowest plate
        for plate_name in self.ordered_plate_dims[::-1]:
            tensors = self.sum_plate_T(tensors, plate_name)

        assert 1==len(tensors)
        lp = tensors[0]
        assert 1==lp.numel()
        return lp

    def sum_plate_T(self, lps, plate_dim): 
        if plate_dim is not None:
            #partition tensors into those with/without plate_name
            lower_lps, higher_lps = partition_tensors(lps, plate_dim)
        else:
            lower_lps = lps
            higher_lps = []

        #collect K's that appear in higher plates
        Ks_to_keep = set([dim for dim in unify_dims(higher_lps) if self.is_K(dim)])

        n_timeseries = sum(isinstance(lp, TimeseriesLogP) for lp in lower_lps)
        assert n_timeseries in [0, 1]
        if n_timeseries == 1:
            lower_lp = self.sum_T(lower_lps, plate_dim, Ks_to_keep)
        else:
            lower_lp = self.sum_plate(lower_lps, plate_dim, Ks_to_keep)

        return [*higher_lps, lower_lp]

    def sum_T(self, lower_lps, T, Ks_to_keep):
        ts     =  next(lp for lp in lower_lps if     isinstance(lp, TimeseriesLogP))
        non_ts = tuple(lp for lp in lower_lps if not isinstance(lp, TimeseriesLogP))
        assert ts.T is T

        #Split all the non-timeseries log-p's into the first and rest.
        non_ts_T = [lp.order(T)    for lp in non_ts]
        firsts   = [lp[0]          for lp in non_ts_T]
        rests    = [lp[1:][ts.Tm1] for lp in non_ts_T]

        #Reduce over Ks separately for first and rest
        Ks_to_keep = [ts.K, *Ks_to_keep]
        first = self.reduce_Ks_to_keep([ts.first, *firsts], Ks_to_keep)
        rest  = self.reduce_Ks_to_keep([ts.rest,  *rests],  Ks_to_keep)
        
        first = first.order(ts.K)[ts.Kprev] #Replace K with Kprev
        rest = chain_logmmmeanexp(rest, ts.Tm1, ts.Kprev, ts.K) #Kprev x Knext

        return reduce_Ks([first, rest], [ts.Kprev, ts.K])

    def sum_plate(self, lower_lps, plate_dim, Ks_to_keep):
        lower_lp = self.reduce_Ks_to_keep(lower_lps, Ks_to_keep)
        if plate_dim is not None:
            lower_lp = lower_lp.sum(plate_dim)
        return lower_lp

    def reduce_Ks_to_keep(self, tensors, Ks_to_keep):
        """
        Takes a list of log-probability tensors, and a list of dimension names to do the reductions,
        and does a numerically stable log-einsum-exp

        Arguments:
            tensors: List of tensors within a plate
            Ks_to_keep: List of K_dims to keep because they appear in higher-level plates.
        Returns: a single log-probability tensor with all K's appearing only in this plate summed out

        Same for timeseries and plate!
        """

        all_dims = unify_dims(tensors)
        Ks_to_keep = set(Ks_to_keep)
        Ks_to_sum    = [dim for dim in all_dims if self.is_K(dim) and (dim not in Ks_to_keep)]
        return reduce_Ks(tensors, Ks_to_sum)


    def elbo(self):
        return self.tensor_product()

    def rws(self):
        # Wake-phase P update
        p_obj = self.tensor_product(detach_q=True)
        # Wake-phase Q update
        q_obj = self.tensor_product(detach_p=True)
        return p_obj, q_obj

    def weights(self):
        """
        Produces normalized weights for each latent variable.

        Make a little function that converts all the aunnamed to dim tensors
        """
        var_names     = list(self.trp.samples.keys())
        samples       = list(self.trp.samples.values())
        logqs         = [self.trp.logq[var_name] for var_name in var_names]
        dimss         = [lq.dims for lq in logqs]
        undim_logqs   = [generic_order(lq, dims) for (lq, dims) in zip(logqs, dimss)]

        #Start with Js with no dimensions (like NN parameters)
        undim_Js = [t.zeros_like(ulq, requires_grad=True) for ulq in undim_logqs]
        #Put torchdims back in.
        dim_Js = [J[dims] for (J, dims) in zip(undim_Js, dimss)]
        #Compute result with torchdim Js
        result = self.tensor_product(extra_log_factors=dim_Js)
        #But differentiate wrt non-torchdim Js
        ws = list(t.autograd.grad(result, undim_Js))

        result = {}
        for i in range(len(ws)):
            sample = dim2named_tensor(samples[i], dimss[i])
            w = ws[i].rename(*[repr(dim) for dim in dimss[i]])
            #Replace arbitrary K with general K.
            K_dim = next(dim for dim in dimss[i] if self.is_K(dim))
            K_name = repr(K_dim)
            K_dict = {K_name: 'K'}
            sample = sample.rename(**K_dict)
            w      = w.rename(**K_dict)

            result[var_names[i]] = (sample, w)

        return result

    def _importance_samples(self, N):
        """
        Returns torchdim tensors, so not for external use.
        """
        #ordered in the order of generating under P
        var_names     = list(self.trp.samples.keys())
        samples       = list(self.trp.samples.values())
        logps         = [self.trp.logp[var_name] for var_name in var_names]
        dimss         = [lp.dims for lp in logps]
        undim_logps   = [generic_order(lp, dims) for (lp, dims) in zip(logps, dimss)]

        #Start with Js with no dimensions (like NN parameters)
        undim_Js = [t.zeros_like(ulp, requires_grad=True) for ulp in undim_logps]
        #Put torchdims back in.
        dim_Js = [J[dims] for (J, dims) in zip(undim_Js, dimss)]
        #Compute result with torchdim Js
        result = self.tensor_product(extra_log_factors=dim_Js)
        #But differentiate wrt non-torchdim Js
        marginals = list(t.autograd.grad(result, undim_Js))
        #Put dims back,
        marginals = [marg[dims] for (marg, dims) in zip(marginals, dimss)]
        #Normalized, marg gives the "posterior marginals" over Ks

        Ks_so_far = set()
        #Dict mapping Kdim to NxPlates indexes.
        K_post_idxs = {}
        post_samples = {}

        #Go through each variable in the order it was generated in P
        for i in range(len(var_names)):
            #All Ks in this variable.
            Ks = [dim for dim in dimss[i] if self.is_K(dim)]
            #Split into Ks that are new for this variable, and old
            new_Ks  = [dim for dim in Ks if (dim not in Ks_so_far)]
            prev_Ks = [dim for dim in Ks if (dim     in Ks_so_far)]
            #Should be zero (e.g. if grouped) or one new K.
            assert len(new_Ks) in (0, 1)

            #If there's a new K, then we need to do posterior sampling for that K.
            if 1==len(new_Ks):
                K = new_Ks[0]
                Ks_so_far.add(K)
                #index into marg for the previous Ks, which gives an unnormalized posterior.
                marg = generic_order(marginals[i], prev_Ks)
                cond = marg[tuple(K_post_idxs[prev_K] for prev_K in prev_Ks)]

                #Check none of the conditional probabilites are big and negative
                assert (-1E-6 < generic_order(cond, generic_dims(cond))).all()
                #Set any small and negative conditional probaiblities to zero.
                cond = cond * (cond > 0)

                #Put current K at the back for the sampling,
                cond = cond.order(K)
                cond = cond.permute(cond.ndim-1, *range(cond.ndim-1))
                #Sample new K's
                K_post_idxs[K] = Categorical(cond).sample(False, sample_dims=(N,))

            sample_K = next(dim for dim in samples[i].dims if self.is_K(dim))
            post_samples[var_names[i]] = samples[i].order(sample_K)[K_post_idxs[sample_K]]
        return post_samples

    def _importance_samples(self, N):
        """
        Returns torchdim tensors, so not for external use.
        """
        #ordered in the order of generating under P
        var_names           = list(self.trp.samples.keys())
        samples             = list(self.trp.samples.values())
        logps_u             = [self.trp.logp[var_name] for var_name in var_names]
        #Some of the objects in logps_u aren't tensors!  They're TimeseriesLogP, which acts as a
        #container for a first and last tensor.  To take gradients we need actual tensors, so we flatten,
        logps_f, unflatten  = flatten_tslp_list(logps_u)
        dimss               = [lp.dims for lp in logps_f]
        undim_logps         = [generic_order(lp, dims) for (lp, dims) in zip(logps_f, dimss)]

        #Start with Js with no dimensions (like NN parameters)
        undim_Js = [t.zeros_like(ulp, requires_grad=True) for ulp in undim_logps]
        #Put torchdims back in.
        dim_Js = [J[dims] for (J, dims) in zip(undim_Js, dimss)]
        #Compute result with torchdim Js
        result = self.tensor_product(extra_log_factors=dim_Js)
        #But differentiate wrt non-torchdim Js
        marginals = list(t.autograd.grad(result, undim_Js))
        #Put dims back,
        marginals = [marg[dims] for (marg, dims) in zip(marginals, dimss)]
        #Normalized, marg gives the "posterior marginals" over Ks

        #Wrap back up the first and last marginals relating to TimeseriesLogP.
        marginals = unflatten(marginals)

        Ks_so_far = set()
        #Dict mapping Kdim to NxPlates indexes.
        K_post_idxs = {}
        post_samples = {}

        #Go through each variable in the order it was generated in P
        for i in range(len(var_names)):
            #All Ks in this variable.
            Ks = [dim for dim in logps_u[i].dims if self.is_K(dim)]
            #Ks = [dim for dim in  if self.is_K(dim)]
            #Split into Ks that are new for this variable, and old
            new_Ks  = [dim for dim in Ks if (dim not in Ks_so_far)]
            prev_Ks = [dim for dim in Ks if (dim     in Ks_so_far)]
            #Should be zero (e.g. if grouped) or one new K.
            assert len(new_Ks) in (0, 1)

            #If there's a new K, then we need to do posterior sampling for that K.
            if 1==len(new_Ks):
                K = new_Ks[0]
                Ks_so_far.add(K)
                #index into marg for the previous Ks, which gives an unnormalized posterior.
                marg = generic_order(marginals[i], prev_Ks)
                cond = marg[tuple(K_post_idxs[prev_K] for prev_K in prev_Ks)]

                #Check none of the conditional probabilites are big and negative
                assert (-1E-6 < generic_order(cond, generic_dims(cond))).all()
                #Set any small and negative conditional probaiblities to zero.
                cond = cond * (cond > 0)

                #Put current K at the back for the sampling,
                cond = cond.order(K)
                cond = cond.permute(cond.ndim-1, *range(cond.ndim-1))
                #Sample new K's
                K_post_idxs[K] = Categorical(cond).sample(False, sample_dims=(N,))

            sample_K = next(dim for dim in samples[i].dims if self.is_K(dim))
            post_samples[var_names[i]] = samples[i].order(sample_K)[K_post_idxs[sample_K]]
        return post_samples
