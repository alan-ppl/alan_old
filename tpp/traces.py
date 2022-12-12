from .tensor_product import Sample

import torch as t
import torch.nn as nn
from functorch.dim import dims, Dim

from .utils import *

def insert_size_dict(d, size_dict):
    new_dict = {}
    for (name, size) in size_dict.items():
        if (name not in d):
            new_dict[name] = Dim(name, size)
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
    if 0==x.ndim:
        return x

    torchdims = [(slice(None) if (name is None) else d[name]) for name in x.names]
    return x.rename(None)[torchdims]

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
    plates = insert_named_tensors(plates, named_data.values())

    #Convert data named tensors to torchdim tensors
    dim_data = {k: named2dim_tensor(plates, tensor) for (k, tensor) in named_data.items()}
    return dim_data, plates

class Q(nn.Module):
    """
    Key problem: parameters in Q must have torchdim plates.
    Solve this problem by making a new method to register parameters, "reg_param", which takes 
    a named tensor, and builds up a mapping from names to torchdims.
    """
    def __init__(self):
        super().__init__()
        self._plates = {}
        self._params = nn.ParameterDict()
        self._dims = {}

    def reg_param(self, name, tensor, dims=None):
        """
        Tensor could be named, or we could provide a dims (iterable of strings) argument.
        """
        #Save unnamed parameter
        self._params[name] = nn.Parameter(tensor.rename(None))

        #Put everything into names, and generate names for each dim.
        if dims is not None:
            assert tensor.names == tensor.ndim*(None,)
            tensor = tensor.rename(*dims, *((tensor.ndim - len(dims))*[None]))
        self._plates = insert_named_tensor(self._plates, tensor)

        tensor_dims = []
        for dimname in tensor.names:
            if dimname is None:
                tensor_dims.append(slice(None))
            else:
                tensor_dims.append(self._plates[dimname])
        if 0==tensor.ndim:
            tensor_dims.append(Ellipsis)
        self._dims[name] = tensor_dims

    def __getattr__(self, name):
        if name == "_params":
            return self.__dict__["_modules"]["_params"]
        else:
            return self._params[name][self._dims[name]]

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

    def sample(self, K, reparam, data):
        data, plates = named2dim_data(data, self.plates)
        all_data = {**self.data, **data}
        assert len(all_data) == len(self.data) + len(data)
       
        #sample from approximate posterior
        trq = TraceQ(K, all_data, plates, reparam)
        self.Q(trq)
        #compute logP
        trp = TraceP(trq)
        self.P(trp)

        return Sample(trp)

    def elbo(self, K, data=None):
        return self.sample(K, True, data).elbo()

    def rws(self, K, data=None):
        return self.sample(K, False, data).rws()

    def weights(self, K, data=None):
        return self.sample(K, False, data).weights()

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

def sample(P, sizes=None, varnames=None):
    if sizes is None:
        sizes = {}
    tr = TraceSample(sizes)
    P(tr)

    if varnames is None:
        varnames = tr.samples.keys()

    return {varname: dim2named_tensor(tr.samples[varname]) for varname in varnames}


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
            self.Ks.add(Kdim)

            sample_q = self.trq[key]
            sample = sample_q.order(self.trq.Kdim)[Kdim]
            self.samples[key] = sample

            self.logq[key] = self.trq.logq[key].order(self.trq.Kdim)[Kdim]

        self.logp[key] = dist.log_prob(sample)
