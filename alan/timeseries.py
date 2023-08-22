import torch as t
from .utils import *
import functorch
from functorch.dim import Dim

Tensor = (t.Tensor, functorch.dim.Tensor)

class Timeseries():
    """
    Requires an initial state which has one and only one K-dimension
    That can be guaranteed if we have directly sampled this initial
    state from a distribution (such as a Gaussian).
    """
    def __init__(self, initial_state_key, transition, inputs=()):
        assert isinstance(initial_state_key, str)
        self.initial_state_key = initial_state_key
        self.transition = transition
        self.init_inputs(inputs)

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

    def init_inputs(self, inputs):
        if isinstance(inputs, Tensor):
            inputs = (inputs,)
        self._inputs = inputs

    @classmethod
    def pred_init(cls, initial_state, transition, Tdim, inputs=()):
        """
        Only called when making predictions.  This is actually the
        simple + obvious version of the constructor.  We do something
        more complex as standard because we don't know e.g. Tdim when
        initializing.
        """
        self = object.__new__(cls)
        self.initial_state = initial_state
        self.transition = transition
        self.Tdim = Tdim
        self.init_inputs(inputs)
        return self

    def inputs(self, t):
        return tuple(x.order(self.Tdim)[t] for x in self._inputs)

    def sample(self, reparam, sample_dims, Kdim=None):
        #Passing in a Kdim indicates that we're in Q, and hence that we
        #need to select parent particles (see index).
        #Kdim doesn't make sense in P.
        index = Kdim is not None
        Knext = (Dim('Knext', self.trace.K),) if index else ()
        sample_dims = (*sample_dims, *Knext)

        #self.initial_sample and hence sample has arbitrary K's
        sample = self.transition(self.initial_state, *self.inputs(0)).sample(reparam, sample_dims)
        if index:
            sample = self.trace.index_sample(sample, *Knext, None)
            sample = sample.order(Knext)[Kdim]
        result = [sample]

        for _t in range(1, self.Tdim.size):
            sample = self.transition(result[-1], *self.inputs(_t)).sample(reparam, sample_dims=Knext)
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
        first = self.transition(self.initial_state, *self.inputs(0)).log_prob(x_first)

        inputs_rest = self.inputs(slice(1, None))
        inputs_rest = tuple(x[Tm1] for x in inputs_rest)
        rest  = self.transition(x_prev, *inputs_rest).log_prob(x_curr)

        return t.cat([first[None], rest.order(Tm1)], 0)[T]
