import torch as t
import torch.nn as nn
from torch.autograd import grad
from .prob_prog import TraceSample, TraceSampleLogQ, TraceLogP
from .inference import logPtmc
from .utils import *
from .tensor_utils import dename, sum_none_dims, nameify
from .backend import is_K

def zeros_like_noK(x, requires_grad=False):
    names = tuple(name for name in x.names if not is_K(name))
    shape = tuple(x.size(name) for name in names)
    return t.zeros(shape, names=names, dtype=x.dtype, device=x.device, requires_grad=requires_grad)



class Model(nn.Module):
    def __init__(self, P, Q, data=None):
        super().__init__()
        self.P = P
        self.Q = Q
        self.data = data

    def traces(self, K, reparam):
        K_dim = Dim(name='K', size=K)
        #sample from approximate posterior
        trq = TraceSampleLogQ(data=self.data, reparam=reparam, K_dim=K_dim)
        self.Q(trq)
        #compute logP
        trp = TraceLogP(trq, self.data, K_dim=K_dim)
        self.P(trp)
        return trp, trq

    def elbo(self, K):
        trp, trq = self.traces(K, reparam=True)
        return logPtmc(trp.logp, trq.logp)

    def rws(self, K):
        trp, trq = self.traces(K, reparam=False)
        # Wake-phase P update
        p_obj = logPtmc(trp.logp, {n:lq.detach() for (n,lq) in trq.logp.items()})
        # Wake-phase Q update
        q_obj = logPtmc({n:lp.detach() for (n,lp) in trp.logp.items()}, trq.logp)
        return p_obj, q_obj

    def moments(self, K, d):
        """
        d is a dict, mapping strings representing the moment_name to tuples of
        the rv_name and a function (e.g. square for the second moment).
        moment_name: (rv_name, function)
        """
        trp, trq = self.traces(K, reparam=False)
        #Convert d to list
        d = [(k, rvn, f) for (k, (rvn, f)) in d.items()]

        extra_log_factors = []
        Js = []
        for (k, rvn, f) in d:
            sample = trp.sample[rvn]
            sample, _, _ = nameify(sample)
            sample = sample[0]
            m = f(sample)
            J = zeros_like_noK(sample, requires_grad=True)
            Js.append(J)
            extra_log_factors.append(m*J.align_as(m))

        result = logPtmc(trp.logp, trq.logp, extra_log_factors)

        Ems = t.autograd.grad(result, Js)
        #Convert back to dict:
        return {k: Em for ((k, _, _), Em) in zip(d, Ems)}


    def pred_likelihood(self, test_data, num_samples, reweighting=False, reparam=True):
        K_dim = Dim(name='K', size=1)
        pred_lik = 0
        #gotta be able to parallelise this
        for i in range(num_samples):
            trq = TraceSampleLogQ(data=test_data, reparam=reparam, K_dim=K_dim)
            self.Q(trq)
            # print(trq.sample)
            #compute logP
            trp = TraceLogP(trq, test_data, K_dim=K_dim)
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
