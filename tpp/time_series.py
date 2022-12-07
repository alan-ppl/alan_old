import torch.distributions as td
from functorch.dim import dims, Dim

class TimeSeries:
    """
    Behaves a bit like a TorchDimDist, but log-prob is way more complicated:

    Problem 1: T-1 transitions, so T-1 elements in log_prob, but T samples in x.  
    So we can't use the same torchdim for the T time-dimension in x and the T-1 time dimension in log-prob.
    Solution: explicitly pass in a Tm1 torchdim.

    Problem 2: We have only a single K-dimension on the inputs, so we have only a single K-dimension on the log-prob (naively)
    But we need two K-dimensions for logP.
    Solution: explicitly pass in Kprev.
    """
    def __init__(self, initial_state, transitions, T):
        self.initial_state = initial_state
        self.transitions = transitions
        assert isinstance(T, Dim)
        self.T = T

    def sample(self, reparam, sample_dims=()):
        result_list = []
        result_list.append(self.transitions(self.initial_state).sample(reparam, sample_dims))
        for t in range(self.T.size-1)
            result_list.append(self.transitions(states[-1]).rsample(reparam)
        result_tensor = t.stack(result_list, 0)
        return result_tensor[T]

    def log_prob_init(self, x):
        x_tensor = x.order(self.T) 
        return self.transition(self.initial_state).log_prob(x_tensor[0])

    def log_prob_rest_P(self, x, Tm1):
        x_tensor = x.order(self.T) 
        prev_state = x_tensor[:-1][Tm1]
        curr_state = x_tensor[1:][Tm1]

        return self.transition(prev_state).log_prob(curr_state)

    def log_prob_rest_Q(self, x, Tm1, K, Kprev):
        x_tensor = x.order(self.T, K) 

        prev_state = x_tensor[:-1][Tm1, Kprev]
        curr_state = x_tensor[1:][Tm1, K]
        return self.transition(prev_state).log_prob(curr_state)
