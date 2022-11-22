import torch.nn as nn
from .prob_prog import TraceSample, TraceSampleLogQ, TraceLogP
from .backend import vi, reweighted_wake_sleep, gibbs, sum_lps, sum_logpqs
# from .cartesian_tensor import CartesianTensor
from .utils import *
from .tensor_utils import dename, sum_none_dims

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
        # print('P log prob')
        # print(trp.log_prob())
        # print('sample')
        # print(trp.sample)
        # print('Q log prob')
        # print(trq.log_prob())
        return vi(trp.log_prob(), trq.log_prob(), dims)

    def importance_sample(self, dims):
        #sample from approximate posterior
        trq = TraceSampleLogQ(dims=dims, data=self.data)
        self.Q(trq)
        #compute logP
        trp = TraceLogP(trq.sample, self.data, dims=dims)
        self.P(trp)
        _, marginals = sum_logpqs(trp.log_prob(), trq.log_prob(), dims)
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

    def pred_likelihood(self, dims, test_data, num_samples, reweighting=False, reparam=True):
        pred_lik = 0
        #gotta be able to parallelise this
        for i in range(num_samples):
            trq = TraceSampleLogQ(dims=dims, data=test_data, reparam=reparam)
            self.Q(trq)
            # print(trq.sample)
            #compute logP
            trp = TraceLogP(trq.sample, test_data, dims=dims)
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
