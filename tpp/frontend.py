import torch.nn as nn
from .prob_prog import TraceSample, TraceSampleLogQ, TraceLogP
from .backend import sum_lps, sum_logpqs
from .inference import vi, reweighted_wake_sleep, gibbs
# from .cartesian_tensor import CartesianTensor
from .utils import *
from .tensor_utils import dename, sum_none_dims

class Model(nn.Module):
    def __init__(self, P, Q, data=None):
        super().__init__()
        self.P = P
        self.Q = Q
        self.data = data

    def elbo(self, K):
        K_dim = Dim(name='K', size=K)
        #sample from approximate posterior
        trq = TraceSampleLogQ(dims=dims, data=self.data, K_dim=K_dim)

        self.Q(trq)
        #compute logP
        trp = TraceLogP(trq, self.data, K_dim=K_dim)
        self.P(trp)
        return vi(trp.log_prob(), trq.log_prob(), trp.dims)

    def importance_sample(self, K):
        K_dim = Dim(name='K', size=K)
        #sample from approximate posterior
        trq = TraceSampleLogQ(data=self.data, K_dim=K_dim)
        self.Q(trq)
        #compute logP
        trp = TraceLogP(trq, self.data, K_dim=K_dim)
        self.P(trp)
        _, marginals = sum_logpqs(trp.log_prob(), trq.log_prob(), trp.dims)
        return gibbs(marginals)

    def rws(self, K):
        K_dim = Dim(name='K', size=K)
        #sample from approximate posterior
        trq = TraceSampleLogQ(data=self.data, reparam=False, K_dim=K_dim)
        self.Q(trq)
        #compute logP
        # trq.sample = {k:v.detach() for k,v in trq.sample.items()}
        trp = TraceLogP(trq, self.data, K_dim=K_dim)
        self.P(trp)

        return reweighted_wake_sleep(trp.log_prob(), trq.log_prob(), trp.dims)

    def pred_likelihood(self, test_data, num_samples, reweighting=False, reparam=True):
        K_dim = Dim(name='K', size=1)
        pred_lik = 0
        #gotta be able to parallelise this
        for i in range(num_samples):
            trq = TraceSampleLogQ(data=test_data, reparam=reparam, K_dim=K_dim)
            self.Q(trq)
            # print(trq.sample)
            #compute logP
            trp = TraceLogP(trq.sample, test_data, K_dim=K_dim)
            self.P(trp)
            # print(trp.log_prob())
            logps = {rv: sum_none_dims(lp) for (rv, lp) in trp.log_prob().items()}
            # print(list(logps.values()))
            # print(logps['obs'])
            # print(dename(logps['obs']).shape)
            # print(logps['obs'].sum())
            pred_lik += logps['obs'].sum()
        shape = dename(test_data['obs']).shape
        # pred_lik = pred_lik.rename(None).reshape(shape, -1)
        return pred_lik / num_samples

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

def make_dims(P, K):
    tr = TraceSample()
    P(tr)

    groups = {}
    for v in set(tr.groups.values()):
        groups[v] = Dim(name='K_{}'.format(v), size=K)
    dims = {'K':Dim(name='K', size=K)}
    for k,v in tr.groups.items():
        dims[k] = groups[v] if tr.groups[k] is not None else Dim(name='K_{}'.format(k), size=K)

    return dims
