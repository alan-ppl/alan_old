import torch.nn as nn
from .prob_prog import TraceSample, TraceSampleLogQ, TraceLogP
from .backend import vi, gibbs

class Model(nn.Module):
    def __init__(self, P, Q, data=None):
        super().__init__()
        self.P = P
        self.Q = Q
        self.data = data

    def elbo(self, K):
        #sample from approximate posterior
        trq = TraceSampleLogQ(K=K, data=self.data)
        self.Q(trq)
        #compute logP
        trp = TraceLogP(trq.sample, self.data)
        self.P(trp)
        return vi(trp.log_prob(), trq.log_prob())


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
