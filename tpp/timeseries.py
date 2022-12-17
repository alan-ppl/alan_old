import torch as t
from .utils import *
from functorch.dim import Dim

class Timeseries():
    def __init__(self, initial_state, transition, inputs=None):
        self.initial_state = initial_state 
        self.transition = transition 
        self._inputs = inputs

    def set_Tdim(self, Tdim):
        assert isinstance(Tdim, Dim)
        self.Tdim = Tdim

        if self._inputs is not None:
            self._inputs = self._inputs.order(Tdim)

    def input(self, t):
        return () if (self._inputs is None) else (inputs[t],)

    def sample(self, reparam, sample_dims):
        result = [self.transition(self.initial_state, *self.input(0)).sample(reparam, sample_dims)]
        for _t in range(1, self.Tdim.size):
            result.append(self.transition(result[-1], *self.input(_t)).sample(reparam, sample_dims=()))

        #A bug with stacking torchdim tensors.  Should be able to do:
        #return t.stack(result, 0)[self.Tdim]
        #where result has some torchdim tensors. 

        #Actually we need to strip the torchdims before stacking
        result_dims = generic_dims(result[0])
        result = [generic_order(x, result_dims) for x in result]
        return t.stack(result, 0)[[self.Tdim, *result_dims]]

    def log_prob(self, x):
        #No dimensions in initial state that don't also appear in x.
        #Should be the case whenever we're doing "non-tensorised" sampling.
        assert 0==len(set(generic_dims(self.initial_state)).difference(generic_dims(x)))

        Tm1 = Dim('Tm1', self.Tdim.size-1)

        x = x.order(self.Tdim)
        x_first = x[0]
        x_prev  = x[:-1][Tm1]
        x_curr  = x[1:][Tm1]

        lp_first = self.transition(self.initial_state, *self.input(0)).log_prob(x_first)
        inputs_rest = self.input(slice(1, None))
        if inputs_rest != ():
            inputs_rest = (inputs_rest[Tm1],)
        lp_rest  = self.transition(x_prev, *inputs_rest).log_prob(x_curr)

        return t.cat([lp_first[None, ...], lp_rest.order(Tm1)], 0)[self.Tdim]

