from collections import namedtuple

import torch as t
import torch.nn as nn
from functorch.dim import dims, Dim

from .utils import *

def insert_size_dict(d, size_dict):
    if size_dict is None:
        size_dict = {}
    new_dict = {}
    for (name, size) in size_dict.items():
        if (name not in d):
            dim = Dim(name, size)
            new_dict[name] = dim
        else:
            assert size == d[name].size
    return {**d, **new_dict}

def insert_named_tensor(d, tensor):
    """
    Operates on dict mapping string (platename) to Dim (platedim)
    """
    return insert_size_dict(d, {name: tensor.size(name) for name in tensor.names if name is not None})

def insert_named_tensors(d, tensors):
    """
    Operates on dict mapping string (platename) to Dim (platedim)
    """
    for tensor in tensors:
        d = insert_named_tensor(d, tensor)
    return d

def named2dim_tensor(d, x):
    """
    Operates on dict mapping string (platename) to Dim (platedim)
    """
    torchdims = [slice(None) if (name is None) else d[name] for name in x.names]
    return x.rename(None)[torchdims]

def dim2named_tensor(x):
    """
    Doesn't need side information.
    Will fail if duplicated dim names passed in
    """
    dims = generic_dims(x)
    names = [repr(dim) for dim in dims]
    return x[dims].rename(*names, ...)

def named2dim_data(named_data, plates):
    """
    Converts data named tensors to torchdim tensors, and records any plates
    Arguments:
      named_data: dict mapping varname to named tensor data
      plates: dict mapping platename to plate dim
    Returns:
      dim_data: dict mapping varname to torchdim tensor data
      plates: dict mapping platename to plate dim
    """
    #Data often defaults to None.
    if named_data is None:
        named_data = {}
    if plates is None:
        plates = {}

    #Insert any dims in data tensors into plates
    plates = insert_named_tensors(plates, data.values())

    #Convert data named tensors to torchdim tensors
    dim_data = {k: named2dim_tensor(plates, tensor) for (k, tensor) in data.items()}
    return dim_data, plates

class Q(nn.Module):
    """
    Key problem: parameters in Q must have torchdim plates.
    Solve this problem by making a new method to register parameters, "reg_param", which takes 
    a named tensor, and builds up a mapping from names to torchdims.
    """
    def __init__(self):
        super().__init__()
        self._plates = NamesDims()
        self.params = nn.ParameterList()

    def reg_param(self, name, tensor, dims=None):
        """
        Tensor could be named, or we could provide a dims (iterable of strings) argument.
        """
        if dims is not None:
            tensor = tensor.rename(*dims, *((tensor.ndim - len(dims))*[None]))
        self._plates = insert_named_tensor(self._plates, tensor)
        self.params.append(nn.Parameter(tensor.rename(None)))
        setattr(self, name, _plates.named2dim_tensor(_plates, tensor))

class Q_(Q):
    def __init__(self):
        super().__init__()
        self.reg_param('a', t.ones(3,3), dims=("plate_1",))

class Model(nn.Module):
    """
    Plate dimensions come from data.
    Model(P, Q, data) is for non-minibatched data. 
    elbo(K=10, data) is for minibatched data. 

    data is stored as torchdim
    """
    def __init__(self, P, Q, data=None):
        super().__init__()
        self.P = P
        self.Q = Q

        
        if data is None:
            data = {}
        self.data, self.plates = named2dim_data(data, Q._plates)

    def traces(self, K, reparam, data):
        data, plates = named2dim_data(data, self.plates)
        all_data = {**self.data, **data}
        assert len(all_data) == len(self.data) + len(data)
       
        #sample from approximate posterior
        trq = TraceQ(K, all_data, plates, reparam)
        self.Q(trq)
        #compute logP
        trp = TraceP(trq)
        self.P(trp)

        return trp

    def elbo(self, K, data=None):
        trp, trq = self.traces(K, True, data)
        return logPtmc(trp.logp, trq.logp)

    def rws(self, K, data=None):
        trp, trq = self.traces(K, reparam, data)
        # Wake-phase P update
        p_obj = logPtmc(trp.logp, {n:lq.detach() for (n,lq) in trq.logp.items()})
        # Wake-phase Q update
        q_obj = logPtmc({n:lp.detach() for (n,lp) in trp.logp.items()}, trq.logp)
        return p_obj, q_obj

    def weights_inner(self, trp, trq):
        """
        Produces normalized weights for each latent variable.
        """
        var_names = list(trp.sample.keys())
        samples = [nameify(trp.sample[var_name])[0] for var_name in var_names]
        Js = [t.zeros_like(trq.logp[var_name], requires_grad=True) for var_name in var_names]

        result = logPtmc(trp.logp, trq.logp, Js)

        ws = list(t.autograd.grad(result, Js))
        #ws from autograd are unnamed, so here we put the names back.
        for i in range(len(ws)):
            ws[i] = ws[i].rename(*Js[i].names)
        return {var_name: (sample, w.align_as(sample)) for (var_name, sample, w) in zip(var_names, samples, ws)}


    def weights(self, K, N=None):
        if N is None:
            trp, trq = self.traces(K, reparam=False)
            return self.weights_inner(trp, trq)
        #else: run multiple iterations. 

class AbstractTrace():
    def __getitem__(self, key):
        in_data   = key in self.data
        in_sample = key in self.samples
        assert in_data or in_sample
        assert not (in_data and in_sample)
        return self.samples[key] if in_sample else self.data[key]

class TraceSample(AbstractTrace):
    """
    Draws samples from P.  Usually just used to sample fake data from the model.
    sizes is a dict mapping platenames to plate sizes
    """
    def __init__(self, sizes=None):
        super().__init__()
        self.plates = insert_size_dict({}, sizes)
        self.reparam = False

        self.data                 = {} 
        self.samples              = {}

    def sample(self, varname, dist, multi_samples=True, plate=None):
        assert varname not in self.samples
            
        sample_dims = [] if plate is None else [self.plates[plate]]
        self.samples[varname] = dist.sample(reparam=self.reparam, sample_dims=sample_dims)

    def trace(self, varnames=None):
        """
        Returns samples as a named dict
        """
        if varnames is None:
            varnames = self.samples.keys()
        return {varname: dim2named_tensor(self.samples[varname]) for varname in varnames}

def sample(P, sizes=None, varnames=None):
    if sizes is None:
        sizes = {}
    tr = TraceSample(sizes)
    P(tr)
    return tr.trace(varnames)


class TraceQ(AbstractTrace):
    """
    Samples a probabilistic program + evaluates log-probability.
    Usually used for sampling the approximate posterior.
    The latents may depend on the data (as in a VAE), but it doesn't make sense to "sample" data.
    Can high-level latents depend on plated lower-layer latents?  (I think so?)
    """
    def __init__(self, K, data, plates, reparam):
        super().__init__()
        self.Kdim = Dim("K", K)
        self.data = data
        self.plates = plates
        self.reparam = reparam

        self.samples = {}
        self.logq = {}

    def sample(self, key, dist, multi_samples=True, plate=None):
        assert key not in self.data
        assert key not in self.samples
        assert key not in self.logq
            
        sample_dims = []
        if plate is not None:
            sample_dims.append(self.plates.name2dim[plate])
        if multi_samples:
            sample_dims.append(self.Kdim)

        sample = dist.sample(reparam=self.reparam, sample_dims=sample_dims)
        if not multi_samples:
            assert self.Kdim not in sample.dims, "Multiple samples are coming into this variable, so we can't stop it giving multiple samples at the output"

        self.samples[key] = sample
        self.logq[key] = dist.log_prob(sample)
        

class TraceP(AbstractTrace):
    def __init__(self, trq):
        self.trq = trq

        self.samples = {}
        self.logp = {}
        self.logq = {}

        self.groupname2dim = {}

        #Get plates from trq
        self.Ks     = set()

    @property
    def data(self):
        return self.trq.data

    def sample(self, key, dist, group=None, plate=None):
        assert key not in self.samples
        assert key not in self.logp
        assert key not in self.var_to_Kname

        #data
        if key in self.data: 
            sample = self.data[key]
        #latent variable
        else: 
            #grouped K's
            if (group is not None): 
                #new group of K's
                if (group not in self.groupname2dim):
                    self.groupname2dim[group] = Dim(f"K_{group}", self.trq.Kdim.size)
                Kdim = self.groupname2dim[group]
                assert Kdim.size == self.trq.Kdim.size
            #non-grouped K's
            else:
                Kdim = Dim(f"K_{key}", self.trq.Kdim.size)
            self.K.add(Kdim)

            sample_q = self.trq[key]
            sample = sample_q.order(self.trq.Kdim)[Kdim]
            self.samples[key] = sample

            self.logq[key] = self.trq.logq[key].order(self.trq.Kdim)[Kdim]

        self.logp[key] = dist.log_prob(sample)

    
