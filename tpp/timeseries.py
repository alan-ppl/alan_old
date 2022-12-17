from utils import *
from functorch.dim import Dim

class Timeseries():
    def __init__(self, initial_state, transition, inputs=None):
        self.initial_state = initial_state 
        self.transition = transition 
        self.Tdim = Tdim

        if inputs is not None:
            inputs = inputs.order(Tdim)
        self._inputs = inputs

    def inputs(self, t):
        return () if (self._inputs is None) else (inputs[t],)

    def sample(self, reparam, sample_dims):
        result = [self.transition(self.initial_state, *self.input(0)).sample(reparam, sample_dims)]
        for t in range(1, self.Tdim.size):
            result.append(self.transition(result[-1], *self.input(t)).sample(reparam))
        return result.stack(0)[self.Tdim]

    def log_prob(self, x):
        #No dimensions in initial state that don't also appear in x.
        #Should be the case whenever we're doing "non-tensorised" sampling.
        assert len(set(generic_dims(initial_state)).difference(generic_dims(x)))

        Tm1 = Dim('Tm1', self.T.size-1)

        x = x.order(self.Tdim)
        x_first = x[0]
        x_prev  = x[:-1][Tm1]
        x_next  = x[1:][Tm1]

        lp_first = self.transition(self.initial_state, *self.input(0)).log_prob(x_first)
        inputs_rest = *self.input(slice(1, None))
        if inputs_rest != ():
            inputs_rest = (inputs_rest[Tm1],)
        lp_rest  = self.transition(x_prev, *inputs_rest).log_prob(x_rest)

        return t.cat([lp_first[None, ...], lp_rest.order[Tm1]], 0)[self.T]

