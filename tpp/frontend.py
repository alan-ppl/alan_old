import torch.nn as nn
from .prob_prog import TraceSample, TraceSampleLogQ, TraceLogP
from .backend import vi, reweighted_wake_sleep, gibbs, sum_lps, sum_logpqs
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

        return vi(trp.log_prob(), trq.log_prob(), dims)

    def importance_sample(self, dims):
        #sample from approximate posterior
        trq = TraceSampleLogQ(dims=dims, data=self.data)
        self.Q(trq)
        #compute logP
        trp = TraceLogP(trq.sample, self.data, dims=dims)
        self.P(trp)
        _, marginals = sum_logpqs(trp.log_prob(), trq.log_prob())
        return gibbs(marginals)

    def rws(self, dims):
        #sample from approximate posterior
        trq = TraceSampleLogQ(dims=dims, data=self.data, reparam=False)
        self.Q(trq)
        #compute logP
        # trq.sample = {k:v.detach() for k,v in trq.sample.items()}
        trp = TraceLogP(trq.sample, self.data, dims=dims)
        self.P(trp)

        return reweighted_wake_sleep(trp.log_prob(), trq.log_prob(), dims)

    # def liw(self, dims):
    #     #sample from approximate posterior
    #     trq = TraceSampleLogQ(dims=dims, data=self.data)
    #     self.Q(trq)
    #     #compute logP
    #     trq.sample = {k:v.detach() for k,v in trq.sample.items()}
    #     trp = TraceLogP(trq.sample, self.data, dims=dims)
    #     self.P(trp)
    #
    #     return local_iw(trp.log_prob(), trq.log_prob())


def sample(P, *names):
    """
    Arguments:
        P: callable taking a trace
        names: list of strings, representing random variables that are data.
    """
    tr = TraceSample()
    P(tr)

    if 0 == len(names):
        return tr.sample
    else:
        return {n: tr.sample[n] for n in names}
