import torch.nn as nn
from .prob_prog import TraceSample, TraceSampleLogQ, TraceLogP
from .backend import vi, reweighted_wake_sleep, gibbs, sum_lps, sum_logpqs, sum_none_dims
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

    def test_log_like(self, dims, test_data):
        trq = TraceSampleLogQ(dims=dims, data=test_data)
        self.Q(trq)
        #compute logP
        trp = TraceLogP(trq.sample, test_data, dims=dims)
        self.P(trp)
        logps = {rv: sum_none_dims(lp) for (rv, lp) in trp.log_prob().items()}
        return sum_lps(list(logps.values()))[0].item()

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
