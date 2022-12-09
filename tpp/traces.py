import torch as t
import torch.nn as nn
from functorch.dim import dims, Dim

class Plates():
    """
    A two way dict name (str) <-> dim (functorch.dim.Dim)
    Functional API
    """
    def __init__(name2dim=None, dim2name=None):
        if name2dim is None:
            self.name2dim = {}
        if dim2name is None:
            self.dim2name = {}

    def insert_size(self, name, size):
        if name in self.name2td:
            assert size == name2dim[name].size
            return self
        else:
            return Plates({name: dim, **self.name2dim},{dim: name, **self.dim2name})

    def insert_size_dict(d):
        for (name, size) in d.items():
            d = self.insert_size(name, size)
        return d

    def insert_named_tensor(d):
        for (name, size) in zip(d.names, d.shape):
            if name is not None:
                d = self.insert_size(name, size)
        return d

    def cat(self, other):
        name2dim = {**self.name2dim, **other.name2dim}
        dim2name = {**self.name2dim, **other.name2dim}
        assert len(name2dim) == len(self.name2dim) + len(other.dim2name)
        assert len(dim2name) == len(self.dim2name) + len(other.dim2name)
        return Plates(name2dim, dim2name)

    def named2dim_tensor(self, x):
        torchdims = [slice(None) if (name is None) else self.name2dim[name] for name in x.names]
        return x.rename(None)[torchdims]

    def dim2named_tensor(self, x):
        names = [slice(None) if (dim is None) else self.dim2name[name] for dim in x.dims]
        return x.rename(*names)

def proc_data(data, plates):
    """
    Adds all names in data named tensors to plates,
    Converts all data from named tensors to torchdim tensors
    """
    #Data often defaults to None.
    if data is None:
        data = {}

    #Insert any dims in data tensors
    for tensor in data.values():
        plates = plates.insert_named_tensor(tensor)

    #Convert data named tensors to torchdim tensors
    result_data = {k: namedtensor_to_tdtensors(plates, tensor) for (k, tensor) in data.items()}
    return result_data, plates

class Q(nn.Module):
    """
    Key problem: parameters in Q must have torchdim plates.
    Solve this problem by making a new method to register parameters, "reg_param", which takes 
    a named tensor, and builds up a mapping from names to torchdims.
    """
    def __init__(self):
        super().__init__()
        self.___plates = {}

    def reg_param(self, name, tensor):
        """
        Tensor is a named tensor for plate / T dimensions
        We collect all the plate dimensions in self._plates
        """
        self.___plates = self.___plates.insert_named_tensor(tensor)

        parameter_name = "_"+name
        self.register_parameter(parameter_name, nn.Parameter(tensor.rename(None)))
        setattr(self, name, namedtensor_to_tdtensor(self.___plates, tensor))

class Q_(Q):
    def __init__(self):
        super().__init__()
        self.reg_param('a', t.ones(3,3).rename('plate_1', None))

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
        self.data, self.plates = proc_data(data, Q.___plates)

    def traces(self, K, reparam, data):
        data, plates = proc_data(data, plates)
        all_data = {**self.data, **data}
        assert len(all_data) == len(self.data) + len(data)
       
        K_dim = Dim(name='K', size=K)
        #sample from approximate posterior
        trq = TraceSampleLogQ(K, data, plates, reparam)
        self.Q(trq)
        #compute logP
        trp = TraceLogP(trq, data, plates)
        self.P(trp)
        return trp, trq

    def elbo(self, K, data=None):
        trp, trq = self.traces(K, reparam, data)
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

class Trace():
    def __getitem__(self, key):
        in_data   = key in self.data
        in_sample = key in self.sample
        assert in_data or in_sample
        assert not (in_data and in_sample)
        return self.sample[key] if in_sample else self.data[key]

class TraceSample(Trace):
    """
    Draws samples from P.  Usually just used to sample fake data from the model.
    """
    def __init__(self, sizes=None):
        super().__init__()
        if sizes is None:
            sizes = {}
        self.plates = Plates().insert_size_dict(sizes)
        self.reparam = False

        self.data                 = {} 
        self.samples              = {}

    def sample(self, varname, dist, multi_samples=True, plate=None):
        assert varname not in self.sample
            
        sample_dims = [] if plate is None else [self.plates[plate]]
        self.samples[varname] = dist.sample(reparam=self.reparam, sample_dims=sample_dims)

    def trace(self, varnames=None):
        """
        Returns samples as a named dict
        """
        if varnames is None:
            varnames = self.samples.keys()
        return {self.plates.named2dim_tensor(self.sample[varname]) for varname in varnames}


class TraceSampleLogQ(Trace):
    """
    Samples a probabilistic program + evaluates log-probability.
    Usually used for sampling the approximate posterior.
    The latents may depend on the data (as in a VAE), but it doesn't make sense to "sample" data.
    Can high-level latents depend on plated lower-layer latents?  (I think so?)
    """
    def __init__(self, K, data, plates, reparam):
        super().__init__()
        self.K= Dim("K", K)
        self.data = data
        self.plates = plates
        self.reparam = reparam

        self.sample = {}
        self.logp = {}

    def sample(self, key, dist, multi_samples=True, plate=None):
        assert key not in self.data
        assert key not in self.sample
        assert key not in self.logp
            
        sample_dims = []
        if plate is not None:
            sample_dims.append(self.plates[plate])
        if multi_samples:
            sample_dims.append(self.Kdim)

        sample = dist.sample(reparam=self.reparam, sample_dims=sample_dims)
        if not multi_samples:
            assert self.Kdim not in sample.dims, "Multiple samples are coming into this variable, so we can't stop it giving multiple samples at the output"

        self.sample[key] = sample
        self.logp[key] = dist.log_prob(sample)
        

class TraceLogP(Trace):
    def __init__(self, trq, data, plates):
        self.trq = trq

        self.sample = {}
        self.logp = {}

        self.Kname_to_Kdim = {}
        self.var_to_Kname = {}

    def sample(self, key, dist, group=None, plate=None):
        assert key not in self.sample
        assert key not in self.logp
        assert key not in self.var_to_Kdim

        Kname = f'K_{key if (group is None) else group}'
        if Kname in self.Kname_to_Kdim:
            Kdim = self.Kname_to_Kdim[Kname]
            assert Kdim.size == trq.K.size
        else:
            Kdim = Dim(Kname, trq.K.size)
        self.var_to_Kname[key] = Kname

        sample_q = trq[key]
        sample = sample_q.order[trq.Kdim][Kdim]

        self.sample[key] = sample
        self.logp[key] = dist.log_prob(sample)
