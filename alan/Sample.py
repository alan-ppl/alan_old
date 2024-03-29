import math
from .utils import *
from .dist import Categorical
from functorch.dim import dims, Tensor, Dim
from . import traces


class Sample():
    """
    Sample object which stores the samples and log probabilites for P and Q, and then performs computations with them (Elbo, importance weights...).
    
    The user will access and use samples as so:
    
     .. highlight:: python
     .. code-block:: python
     
        # Can pick different ways of sampling from the extended state space
        # this one samples children directly from parents (i.e. ancestral sampling)
        samples = model.sample_same(K)
        
        #Evidence lower bound
        elbo = samples.elbo()
        
        #Importance weights
        weights = samples.weights()

    TODO:
      Check that latents (in samples) appear in logps and logqs
      Check that data appears in logps but not logqs
      Check that all dims are something (plate, timeseries, K)
    """
    def __init__(self, trp, lp_dtype, lp_device):
        """
        Args:
            trp (trace): Trace object storing logps, logqs, samples and data
            lp_dtype (torch.dtype): dtype to use for log-probabilities
            lp_device (torch.device): device to use for log-probabilities

        Raises:
            Exception: If more variables are sampled in Q than in P
            Exception: If any variable is sampled in Q but not P
            Exception: If a variable is provided in the data but not specified in P
        """
        self.trp = trp
        for lp in [*trp.logp.values(), *trp.logq_group.values(), *trp.logq_var.values()]:
            assert lp.shape == ()

        Q_non_sum_varnames = [*trp.group, *trp.logq_var]
        Q_sum_varnames = [*Q_non_sum_varnames, *trp.sum_discrete_varnames]

        variables_in_Q_but_not_P = tuple(set(Q_sum_varnames).difference(trp.samples.keys()))
        if 0 != len(variables_in_Q_but_not_P):
            raise Exception(
                f"Variables {variables_in_Q_but_not_P} sampled in Q but not present in P"
            )
        data_with_no_logp = set(trp.data.keys()).difference(trp.logp.keys())
        if 0 != len(data_with_no_logp):
            raise Exception(
                f"Data {data_with_no_logp} provided, but not specified in P"
            )
        assert set(Q_sum_varnames) == set(trp.samples.keys())

        for (rv, lp) in trp.logp.items():
            assert (rv in Q_sum_varnames) or (rv in trp.data)


        #All keys in Q
        for key in Q_non_sum_varnames:

            #check that any rv in logqs is also in logps
            if key not in trp.logp:
                raise Exception(f"The latent variable '{key}' is sampled in Q but not P.")

            lp = trp.logp[key]
            lq = trp.logq_var[key] if (key in trp.logq_var) else trp.logq_group[trp.group[key]]

            # check same plates/timeseries appear in lp and lq
            #lp_notK = [dim for dim in generic_dims(lp) if not self.is_K(dim)]
            #lq_notK = [dim for dim in generic_dims(lq) if not self.is_K(dim)]
            lp_notK = trp.extract_platedims(lp)
            lq_notK = trp.extract_platedims(lq)
            assert set(lp_notK) == set(lq_notK)

        # self.varname2logp = trp.logp
        self.logp = [*trp.logp.values()]
        self.logq = [*trp.logq_group.values(), *trp.logq_var.values()]

        lp_kwargs = {}
        if lp_device is not None:
            lp_kwargs['device'] = lp_device
        self.trp.samples = {k: x.to(**lp_kwargs) for (k, x) in self.trp.samples.items()}

        if lp_dtype is not None:
            lp_kwargs['dtype']  = lp_dtype
        self.logp = [x.to(**lp_kwargs) for x in self.logp]
        self.logq = [x.to(**lp_kwargs) for x in self.logq]

        self.varname2logp = dict(zip(trp.logp.keys(), self.logp))
        #Assumes that self.lps come in ordered
        self.set_platedims = set(trp.platedims.values())
        self.ordered_plate_dims = [dim for dim in unify_dims(self.logp) if self.is_platedim(dim)]
        self.ordered_plate_dims = [None, *self.ordered_plate_dims]

        #Returns Ks + Es (for discrete variables we're summing out), as we need to sum over both!
        self.Ks = set([*self.trp.Ks, *self.trp.Es])
        #Just Es (need to distinguish, because we normalize when summing over Ks, but we don't when summing over Es)
        self.Es = set(self.trp.Es)



    @property
    def samples(self):
        return self.trp.samples

    @property
    def data(self):
        return self.trp.data

    @property
    def reparam(self):
        return self.trp.reparam

    @property
    def device(self):
        return self.trp.device

    @property
    def platedims(self):
        return self.trp.platedims

    def is_platedim(self, dim):
        return dim in self.set_platedims

    def is_K(self, dim):
        return dim in self.Ks

    def tensor_product(self, detach_p=False, detach_q=False, extra_log_factors=()):# fn=lambda x:x):
        """
        Sums over plates, starting at the lowest plate.
        The key exported method.
        """
        """Computes the tensor product:
        
        .. math::
           \tfrac{1}{K^n} \sum_{\k^n} f^x_{\k_{\pa{x}}}(z) \prod_i f^i_{k_i,\k_{\pa{i}}}(z)

        Args:
            detach_p (bool, optional): Whether to detach gradients on logps. Defaults to False.
            detach_q (bool, optional): Whether to detach gradients on logqs. Defaults to False.
            extra_log_factors (tuple, optional): Any extra log factors to add to the tensor product. Defaults to ().

        Returns:
            tensor: The tensor product
        """
        logp = self.logp
        if detach_p:
            logp = [lp.detach() for lp in logp]

        logq = self.logq
        if detach_q:
            logq = [lq.detach() for lq in logq]


        tensors = [*logp, *[-lq for lq in logq], *extra_log_factors]

        ## Convert tensors to Float64
        tensors = [x.to(dtype=t.float64) for x in tensors]

        #iterate from lowest plate
        for plate_name in self.ordered_plate_dims[::-1]:
            tensors = self.sum_plate_T(tensors, plate_name)

        assert 1==len(tensors)
        lp = tensors[0]
        assert 1==lp.numel()

        return lp


    # def HQ(self, detach_q=False):
    #     """
    #     Computes the Entropy: H(Q) = E_Q[-log(Q)]
    #     """
    #     logq = self.logq
    #     if detach_q:
    #         logq = [lq.detach() for lq in logq]

    #     tensors = [*[-lq for lq in logq]]

    #     ## Convert tensors to Float64
    #     tensors = [x.to(dtype=t.float64) for x in tensors]

    #     #iterate from lowest plate
    #     for plate_name in self.ordered_plate_dims[::-1]:
    #         tensors = self.sum_plate_T(tensors, plate_name)

    #     assert 1==len(tensors)
    #     HQ = tensors[0]
    #     assert 1==lp.numel()
    #     return HQ
    

    def sum_plate_T(self, lps, plate_dim):
        r"""Sums over a plate dimension in the tensor product, it will also log-sum-exp over K dimensions that appear only in that plate. 
        This corresponds to taking the product over the plate dimension in the tensor product.
        
        .. warning::
           We are unsure if this function is working correctly, we have noticed that changing the order of log factors in
           the tensor product can change the result.

        Args:
            lps (list): list of log-probability tensors
            plate_dim (string): Name of plate to be summed over

        Returns:
            list: List of log-probability tensors with the plate dimension summed out and any K dims that appear in the plate
            being log-sum-exp'ed over 
        """
        if plate_dim is not None:
            #partition tensors into those with/without plate_name
            lower_lps, higher_lps = partition_tensors(lps, plate_dim)
        else:
            lower_lps = lps
            higher_lps = []

        #collect K's that appear in higher plates
        Ks_to_keep = set([dim for dim in unify_dims(higher_lps) if self.is_K(dim)])

        if plate_dim in self.trp.Tdim2Ks.keys():
            Kprev, Kdim = self.trp.Tdim2Ks[plate_dim]
            Ks_to_keep = [Kdim, *Ks_to_keep]


        lower_lp = self.reduce_Ks_to_keep(lower_lps, Ks_to_keep)

        if plate_dim in self.trp.Tdim2Ks.keys():
            lower_lp = chain_logmmexp(lower_lp, plate_dim, Kprev, Kdim) #Kprev x Knext
            lower_lp = reduce_Ks([lower_lp], [Kdim], self.Es)
        elif plate_dim is not None:
            lower_lp = lower_lp.sum(plate_dim)

        return [*higher_lps, lower_lp]

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
        return reduce_Ks(tensors, Ks_to_sum, self.Es)


    def elbo(self):
        # if not self.reparam and self.warn:
        #     print(f"Calling .elbo() with reparam=False is improper.")
        #     self.warn=False
        return self.tensor_product()


    def rws(self):
        # Wake-phase P update
        p_obj =   self.tensor_product(detach_q=True)
        # Wake-phase Q update
        q_obj = - self.tensor_product(detach_p=True)
        return p_obj, q_obj

    def moments(self, fs):
        """Compute importance weighted moments.

        Arguments:
            fs: iterable containing (f, ["a", "b"]) function, latent name pairs.
            
        Returns:
            list: List of computed moments
        """
        if callable(fs[0]):
            fs = (fs,)

        ms        = [(f(*[self.samples[v] for v in vs]) if isinstance(vs, tuple) else f(self.samples[vs])) for (f, vs) in fs]
        #Keep only platedims.
        dimss     = [[dim for dim in generic_dims(m) if self.is_platedim(dim)] for m in ms]
        sizess    = [[dim.size for dim in dims] for dims in dimss]
        named_Js  = [t.zeros(sizes, dtype=m.dtype, device=m.device, requires_grad=True) for (m, sizes) in zip(ms, sizess)]
        dimss     = [[*dims, Ellipsis] for dims in dimss]
        dim_Js    = [J[dims] for (J, dims) in zip(named_Js, dimss)]
        factors   = [m*J for (m, J) in zip(ms, dim_Js)]

        #Compute result with torchdim Js
        result = self.tensor_product(extra_log_factors=factors)
        #But differentiate wrt non-torchdim Js
        named_Es = list(t.autograd.grad(result, named_Js))

        if callable(fs[0]):
            named_Es = named_Es[0]

        return named_Es

    def Elogq(self):
        r"""
        Uses importance weighting to approximate $E_{P(z|x)}[log Q(z)]$.
        
        Could also implement using importance weights?
        """
        ms = list(self.logq.values())
        #Keep only platedims.
        dimss     = [[dim for dim in generic_dims(m) if self.is_platedim(dim)] for m in ms]
        sizess    = [[dim.size for dim in dims] for dims in dimss]
        named_Js  = [t.zeros(sizes, dtype=m.dtype, device=m.device, requires_grad=True) for (m, sizes) in zip(ms, sizess)]
        dimss     = [[*dims, Ellipsis] for dims in dimss]
        dim_Js    = [J[dims] for (J, dims) in zip(named_Js, dimss)]
        factors   = [m*J for (m, J) in zip(ms, dim_Js)]

        #Compute result with torchdim Js
        result = self.tensor_product(extra_log_factors=factors)
        #But differentiate wrt non-torchdim Js
        Es = list(t.autograd.grad(result, named_Js))

        return sum(E.sum() for E in Es)

    def weights(self):
        """
        Produces normalized importance weights for each latent variable.

        Make a little function that converts all the unnamed to dim tensors
        """
        var_names     = list(self.samples.keys())
        samples       = list(self.samples.values())
        #Will fail if there are no dims (i.e. for unplated variable with multisample=False)
        dimss         = [sample.dims for sample in samples]

        undim_Js = []
        for (sample, dims) in zip(samples, dimss):
            undim_Js.append(t.zeros(tuple(dim.size for dim in dims), device=sample.device, dtype=t.float64, requires_grad=True))

        #Put torchdims back in.
        dim_Js = [J[dims] for (J, dims) in zip(undim_Js, dimss)]
        #Compute result with torchdim Js
        result = self.tensor_product(extra_log_factors=dim_Js)
        #But differentiate wrt non-torchdim Js
        ws = list(t.autograd.grad(result, undim_Js))

        result = {}
        for i in range(len(ws)):
            sample = samples[i]
            w = ws[i][dimss[i]]

            #Change sample, w from dim2named, replacing K_varname with 'K'
            K_dim = next(dim for dim in dimss[i] if self.is_K(dim))
            K_name = repr(K_dim)
            replacement_dict = {K_name: 'K'}

            sample = dim2named_tensor(sample).rename(**replacement_dict)
            w      = dim2named_tensor(w).rename(**replacement_dict)

            result[var_names[i]] = (sample, w)

        return result

    def importance_samples(self, N):
        """
        Returns t.tensors
        """
        N = Dim('N', N)
        return self._importance_samples(N)

    def _importance_samples(self, N):
        """
        Returns torchdim tensors, so not for external use.
        Divided into two parts: computing the marginals over K, and actually sampling
        The marginals are computed using the derivative of log Z again.
        We sample forward, following the generative model.
        """
        assert isinstance(N, Dim)
        #### Computing the marginals
        #ordered in the order of generating under P
        var_names           = list(self.samples.keys())
        samples             = list(self.samples.values())
        logps               = [self.varname2logp[var_name] for var_name in var_names]
        dimss               = [lp.dims for lp in logps]
        undim_logps         = [generic_order(lp, dims) for (lp, dims) in zip(logps, dimss)]

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
        #Delete everything that's not necessary for the rest
        del logps, dimss, undim_logps, undim_Js, dim_Js, result

        #### Sampling the K's

        #Dict mapping Kdim to NxPlates indexes.
        K_post_idxs = {}
        post_samples = {}

        #Go through each variable in the order it was generated in P
        for i in range(len(var_names)):
            marg = marginals[i]
            #marg could be tensor or TimeseriesLogP.  However, TimeseriesLogP defines
            #.dims, and the resulting new_K should make sense for both .first and .last
            new_Ks = [dim for dim in generic_dims(marg) if self.is_K(dim) and (dim not in K_post_idxs)]

            #Should be zero (e.g. if grouped) or one new K.
            assert len(new_Ks) in (0, 1)

            #If there's a new K, then we need to do posterior sampling for that K.
            if 1==len(new_Ks):
                K = new_Ks[0]
                if var_names[i] in self.trp.Tvar2Tdim.keys():
                    Tdim = self.trp.Tvar2Tdim[var_names[i]]
                    Kprev, Kdim = self.trp.Tdim2Ks[Tdim]
                    marg = marg.order(Tdim)
                    # print(var_names[i])
                    # print(marg)
                    #Tensor to record all the K's
                    init_K_post = sample_cond(marg[0], Kdim, K_post_idxs, N)
                    K_posts    = init_K_post[None, ...].expand(Tdim.size)

                    for _t in range(1, Tdim.size):
                        _K_post_idxs = {Kprev: K_posts[_t-1], **K_post_idxs}

                        #rest runs from t=1...T-1, so rest[0] corresponds to time t=1.
                        #could be optimized by indexing into marg with K_post_idxs once.
                        K_posts[_t] = sample_cond(marg[_t-1], Kdim, _K_post_idxs, N)

                    K_post_idxs[K] = K_posts[Tdim]
                else:
                    K_post_idxs[K] = sample_cond(marg, K, K_post_idxs, N)

            #the sample should only have one K, so pick it out and index into it.
            sample_K = next(dim for dim in samples[i].dims if self.is_K(dim))
            post_samples[var_names[i]] = samples[i].order(sample_K)[K_post_idxs[sample_K]]
        return post_samples

def sample_cond(marg, K, K_post_idxs, N):
    """
    Takes the marginal probability (marg), and a dict of indices for the previous K's (K_post_idxs)
    and returns N (torchdim Dim) samples of the current K (torchdim Dim).
    """
    prev_Ks = [dim for dim in generic_dims(marg) if (dim in K_post_idxs)]
    marg = generic_order(marg, prev_Ks)
    #index into marg for the previous Ks, which gives an unnormalized posterior.
    #note that none of the previous Ks have a T dimension, so we don't need to
    #do any time indexing...

    cond = marg[tuple(K_post_idxs[prev_K] for prev_K in prev_Ks)]

    #Check none of the conditional probabilites are big and negative
    assert cond.dtype == t.float64
    assert (-1E-6 < generic_order(cond, generic_dims(cond))).all()
    #Set any small and negative conditional probaiblities to zero.
    cond = cond * (cond > 0)

    #Put current K at the back for the sampling,
    cond = cond.order(K)
    cond = cond.permute(cond.ndim-1, *range(cond.ndim-1))

    #Sample new K's

    ## Shouldn't need to add 1e-10
    return Categorical(cond + 1e-10).sample(False, sample_dims=(N,))

class SampleGlobal(Sample):
    def tensor_product(self, detach_p=False, detach_q=False, extra_log_factors=()):
        """
        Sums over plates, starting at the lowest plate.
        The key exported method.
        """
        logp = self.logp
        if detach_p:
            logp = [lp.detach() for lp in logp.items()]

        logq = self.logq
        if detach_q:
            logq = [lq.detach() for lq in logq.items()]

        tensors = [*logp.values(), *[-lq for lq in logq.values()], *extra_log_factors]

        ## Convert tensors to Float64
        lpqs = sum(self.sum_not_K(x.to(dtype=t.float64)) for x in tensors)
        return lpqs.logsumexp(self.Kdim) - math.log(self.Kdim.size)

    @property
    def Kdim(self):
        return self.trp.trq.Kdim

    def sum_not_K(self, x):
        dims = set(x.dims)
        assert self.Kdim in dims
        dims.remove(self.Kdim)
        return sum_dims(x, dims)
