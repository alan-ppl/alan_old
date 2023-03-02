import torch as t
from .utils import *
from functorch.dim import Dim

class Timeseries():
    """
    Requires an initial state which has one and only one K-dimension
    That can be guaranteed if we have directly sampled this initial
    state from a distribution (such as a Gaussian).
    """
    def __init__(self, initial_state_key, transition, inputs=None):
        assert isinstance(initial_state_key, str)
        self.initial_state_key = initial_state_key
        self.transition = transition 
        self._inputs = inputs

    def set_trace_Tdim(self, trace, Tdim):
        """
        Timeseries is initialized in user code, and the user (and hence the init) doesn't know:
          the length of the timeseries.
          the type of trace (e.g. Categorical vs Permutation vs Same). 
        """
        assert isinstance(Tdim, Dim)
        self.trace = trace
        self.Tdim = Tdim

        self.initial_state = trace.samples[self.initial_state_key]

    def input(self, t):
        if self._inputs is None:
            return ()
        else:
            return (self._inputs.order(self.Tdim)[t],)

    def sample(self, reparam, sample_dims, Kdim=None):
        #Passing in a Kdim indicates that we're in Q, and hence that we
        #need to select parent particles (see index).
        #Kdim doesn't make sense in P.
        index = Kdim is not None
        Knext = (Dim('Knext', self.trace.K),) if index else ()
        sample_dims = (*sample_dims, *Knext)

        #self.initial_sample and hence sample has arbitrary K's
        sample = self.transition(self.initial_state, *self.input(0)).sample(reparam, sample_dims)
        if index:
            sample = self.trace.index_sample(sample, *Knext, None)
            sample = sample.order(Knext)[Kdim]
        result = [sample]

        for _t in range(1, self.Tdim.size):
            sample = self.transition(result[-1], *self.input(_t)).sample(reparam, sample_dims=Knext)
            if index:
                sample = self.trace.index_sample(sample, *Knext, None)
                sample = sample.order(Knext)[Kdim]
            result.append(sample)

        #A bug with stacking torchdim tensors.  Should be able to do:
        #return t.stack(result, 0)[self.Tdim]
        #where result has some torchdim tensors. 

        #Actually we need to strip the torchdims before stacking
        result_dims = generic_dims(result[0])
        result = [generic_order(x, result_dims) for x in result]
        result = t.stack(result, 0)[[self.Tdim, *result_dims]]
        return result

    def log_prob(self, x, Kdim):
        #Set up key dimensions
        T = self.Tdim
        Tm1 = Dim('Tm1', T.size-1)
        K = Kdim

        initial_Ks = self.trace.extract_Kdims(self.initial_state)
        assert 1==len(initial_Ks)
        Kprev = list(initial_Ks)[0]

        x = x.order(T)
        x_first = x[0]
        x_prev  = x[:-1][Tm1]
        x_curr  = x[1:][Tm1]

        x_prev  = x_prev.order(Kdim)[Kprev]

        #Compute log probabilities
        first = self.transition(self.initial_state, *self.input(0)).log_prob(x_first)

        inputs_rest = self.input(slice(1, None))
        if inputs_rest != ():
            inputs_rest = (inputs_rest[Tm1],)
        rest  = self.transition(x_prev, *inputs_rest).log_prob(x_curr)

        return t.cat([first[None], rest.order(Tm1)], 0)[T]

#class TimeseriesLogP():
#    """
#    The initial state in the timeseries is passed in directly,
#    meaning it can have any number of Ks (typically zero, if
#    deterministic init, or one if sampled from a dist).  That
#    means the first transition in the timeseries has a different
#    size logp compared to all the others.
#    """
#    def __init__(self, first, rest, T, Tm1, K, Kprev):
#        assert all(isinstance(x, Dim) for x in [T, Tm1, K, Kprev])
#        assert T.size == Tm1.size+1
#        assert K.size == Kprev.size
#
#        self.first   = first
#        self.rest    = rest
#        self.T       = T
#        self.Tm1     = Tm1
#        self.K       = K
#        self.Kprev   = Kprev
#
#        all_dims = set(generic_dims(first)).intersection(generic_dims(rest))
#        all_dims.add(T)
#        self.dims = list(all_dims.difference([Kprev, Tm1]))
#
#    def similar(self, first, rest):
#        return TimeseriesLogP(first, rest, self.T, self.Tm1, self.K, self.Kprev)
#
#    def to(self, dtype=None, device=None):
#        first = self.first.to(dtype=dtype, device=device)
#        rest  = self.rest.to(dtype=dtype, device=device)
#        return self.similar(first, rest)
#
#    def __add__(self, other):
#        if isinstance(other, float):
#            return self.similar(self.first + other, self.rest + other)
#        elif isinstance(other, TimeseriesLogP):
#            assert set(self.first.dims) == set(other.first.dims)
#            assert set(self.rest.dims)  == set(other.rest.dims)
#            assert self.first.ndim == other.first.ndim
#            assert self.rest.ndim  == other.rest.ndim
#
#            assert self.K     is other.K
#            assert self.Kprev is other.Kprev
#            assert self.T     is other.T
#            assert self.Tm1   is other.Tm1
#            return self.similar(self.first + other.first, self.rest + other.rest)
#        else:
#            raise Exception(f"Don't know how to add {other} to a TimeseriesLogP")
#            
#
#def tslp_to_tuple(tslp):
#    """
#    Takes a TimeseriesLogP, and returns a tuple of first and rest, and a method for going backwards.
#    """
#    return (tslp.first, tslp.rest), tslp.similar
#
#def flatten_tslp_list(ls):
#    length = len(ls)
#    inverses = []
#    result = []
#    for l in ls:
#        if isinstance(l, TimeseriesLogP):
#            (first, rest), inverse = tslp_to_tuple(l)
#            result.append(first)
#            result.append(rest)
#            inverses.append(inverse)
#        else:
#            result.append(l)
#            inverses.append(None)
#
#    def inverse(xs):
#        xs = [*xs]   #Copy list, as we're going to modify it in-place using pop
#        _result = []
#        for inverse in inverses[::-1]: #Pop pulls things off in reverse order
#            if inverse is None:
#                _result.append(xs.pop())
#            else:
#                rest  = xs.pop()
#                first = xs.pop()
#                _result.append(inverse(first, rest))
#        assert 0==len(xs)
#        assert length == len(_result)
#        return _result[::-1] #Resulting list has been reversed
#
#    return result, inverse
