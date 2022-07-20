import torch.nn as nn
from .prob_prog import TraceSample, TraceSampleLogQ, TraceLogP
from .backend import vi, gibbs, sum_lps, sum_logpqs
# from .cartesian_tensor import CartesianTensor
from .utils import *

class Model(nn.Module):
    def __init__(self, P, Q, data=None):
        super().__init__()
        self.P = P
        self.Q = Q
        self.data = data

    def elbo(self, dims):
        #sample from approximate posterior
        trq = TraceSampleLogQ(dims=dims, data=self.data)
        self.Q(trq)
        #compute logP
        trp = TraceLogP(trq.sample, self.data, dims=dims)
        self.P(trp)
        return vi(trp.log_prob(), trq.log_prob())

    def importance_sample(self, dims):
        #sample from approximate posterior
        trq = TraceSampleLogQ(dims=dims, data=self.data)
        self.Q(trq)
        #compute logP
        trp = TraceLogP(trq.sample, self.data)
        self.P(trp)
        _, marginals = sum_logpqs(trp.log_prob(), trq.log_prob())
        return gibbs(marginals)


def sample(P, *names):
    """
    Arguments:
        P: callable taking a trace
        names: list of strings, representing random variables that are data.
    """
    tr = TraceSample()
    P(tr)
    if 0 == len(names):
        return dename(tr.sample)
    else:
        return {n: dename(tr.sample[n]) for n in names}
