from .utils import *
from .dist import Categorical


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
            #check all dimensions are named
            assert lp.shape == ()
            #Check dimensions are Ks or plates it doesn't check if all dimensions are named.
            #for dim in lp.dims:
            #    assert self.is_K(dim) or self.is_plate(dim)


        for (rv, lq) in trp.logq.items():
            #check that any rv in logqs is also in logps
            assert rv in trp.logp

            lp = trp.logp[rv]
            lq = trp.logq[rv]

            # check same plates/timeseries appear in lp and lq
            lp_notK = [dim for dim in lp.names if not self.is_K(dim)]
            lq_notK = [dim for dim in lq.names if not self.is_K(dim)]
            assert set(lp_notK) == set(lq_notK)


        #Assumes that self.lps come in ordered
        self.plates = set(trp.trq.plates.values())
        self.ordered_plate_dims = [dim for dim in unify_dims(trp.logp.values()) if self.is_plate(dim)]
        self.ordered_plate_dims = [None, *self.ordered_plate_dims]

    def is_plate(self, dim):
        return dim in self.plates

    def is_K(self, dim):
        return dim in self.trp.Ks

    def tensor_product(self, logps=None, logqs=None, extra_log_factors=()):
        """
        Sums over plates, starting at the lowest plate. 
        The key exported method.
        """
        if logps is None:
            logps = self.trp.logp
        if logqs is None:
            logqs = self.trp.logq

        #combine all lps, negating logqs
        lpqs = []
        for key in logps:
            lpq = logps[key]
            if key in logqs:
                lpq = lpq - logqs[key]
            lpqs.append(lpq)

        tensors = [*lpqs, *extra_log_factors]

        #iterate from lowest plate
        for plate_name in self.ordered_plate_dims[::-1]:
            tensors = self.sum_plate(tensors, plate_name)

        assert 1==len(tensors)
        lp = tensors[0]
        assert 1==lp.numel()
        return lp

    def sum_plate(self, lps, plate_dim):
        """
        Arguments:
            tensors: list of all tensor factors, both including and not including plate.
            plate_dim
        Returns:
            lps: full list of log-probability tensors with plate_name summed out
        This is the only method that differs for time-series vs plate
        """
        if plate_dim is not None:
            #partition tensors into those with/without plate_name
            lower_lps, higher_lps = partition_tensors(lps, plate_dim)
        else:
            lower_lps = lps
            higher_lps = []

        #collect K's that appear in higher plates
        Ks_to_keep = set([dim for dim in unify_dims(higher_lps) if self.is_K(dim)])

        lower_lp = self.reduce_Ks(lower_lps, Ks_to_keep)
        if plate_dim is not None:
            lower_lp = lower_lp.sum(plate_dim)


        return [*higher_lps, lower_lp]
            
    def reduce_Ks(self, tensors, Ks_to_keep):
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
        Ks_to_sum    = [dim for dim in all_dims if self.is_K(dim) and (dim not in Ks_to_keep)]
        
        #subtract max to ensure numerical stability.  Do max over all dims that we're summing over.
        maxes = [max_dims(tensor, Ks_to_sum) for tensor in tensors]
        tensors_minus_max = [(tensor - m).exp() for (tensor, m) in zip(tensors, maxes)]
        result = torchdim_einsum(tensors_minus_max, Ks_to_sum).log() 
        result = result - len(Ks_to_sum)*t.log(t.tensor(self.trp.trq.Kdim.size))

        return sum([result, *maxes])

    def elbo(self):
        return self.tensor_product()

    def rws(self):
        assert not self.reparam
        # Wake-phase P update
        p_obj = self.tensor_product(logqs={n:lq.detach() for (n,lq) in self.trp.logq.items()})
        # Wake-phase Q update
        q_obj = self.tensor_product(logps={n:lp.detach() for (n,lp) in trp.logp.items()})
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
        #Normalized, marg gives the "posterior marginals" over 
        
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

