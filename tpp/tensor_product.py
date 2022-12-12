from .utils import *


class TensorProduct():
    """
    Does error checking on the log-ps, and does the tensor product.

    TODO:
      Check that latents (in samples) appear in logps and logqs
      Check that data appears in logps but not logqs
      Check that all dims are something (plate, timeseries, K)
    """
    def __init__(self, trp):
        self.trp = trp

        logqs = trp.logq
        logps = trp.logp


        for lp in [*logps.values(), *logqs.values()]:
            #check all dimensions are named
            assert lp.shape == ()
            #Check dimensions are Ks or plates it doesn't check if all dimensions are named.
            #for dim in lp.dims:
            #    assert self.is_K(dim) or self.is_plate(dim)


        for (rv, lq) in logqs.items():
            #check that any rv in logqs is also in logps
            assert rv in logps

            lp = logps[rv]
            lq = logqs[rv]

            # check same plates/timeseries appear in lp and lq
            lp_notK = [dim for dim in lp.names if not self.is_K(dim)]
            lq_notK = [dim for dim in lq.names if not self.is_K(dim)]
            assert set(lp_notK) == set(lq_notK)

        #combine all lps, negating logqs
        self.lpqs = []
        for key in logps:
            lpq = logps[key]
            if key in logqs:
                lpq = lpq - logqs[key]
            self.lpqs.append(lpq)

        #Assumes that self.lps come in ordered
        self.plates = set(trp.trq.plates.values())
        self.ordered_plate_dims = [dim for dim in unify_dims(logps.values()) if self.is_plate(dim)]
        self.ordered_plate_dims = [None, *self.ordered_plate_dims]

    def is_plate(self, dim):
        return dim in self.plates

    def is_K(self, dim):
        return dim in self.trp.Ks

    def __call__(self, extra_log_factors=()):
        """
        Sums over plates, starting at the lowest plate. 
        The key exported method.
        """

        tensors = [*self.lpqs, *extra_log_factors]

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
