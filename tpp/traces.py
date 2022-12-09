import torch as t
import torch.nn as nn
from functorch.dim import dims, Dim
from .namesdims import NamesDims

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
    result_data = {k: plates.named2dim_tensor(tensor) for (k, tensor) in data.items()}
    return result_data, plates

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

    def reg_param(self, name, tensor):
        """
        Tensor is a named tensor for plate / T dimensions
        We collect all the plate dimensions in self._plates
        """
        self._plates = self._plates.insert_named_tensor(tensor)
        self.params.append(nn.Parameter(tensor.rename(None)))
        setattr(self, name, self._plates.named2dim_tensor(tensor))

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
        self.data, self.plates = proc_data(data, NamesDims())

    def traces(self, K, reparam, data):
        data, plates = proc_data(data, self.Q._plates)
        all_data = {**self.data, **data}
        assert len(all_data) == len(self.data) + len(data)
       
        K_dim = Dim(name='K', size=K)
        #sample from approximate posterior
        trq = TraceSampleLogQ(K, all_data, plates, reparam)
        self.Q(trq)
        #compute logP
        trp = TraceLogP(trq)
        self.P(trp)
        return trp, trq

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
    """
    def __init__(self, sizes=None):
        super().__init__()
        if sizes is None:
            sizes = {}
        self.plates = NamesDims().insert_size_dict(sizes)
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
        return {varname: self.plates.named2dim_tensor(self.samples[varname]) for varname in varnames}

def sample(P, sizes=None, varnames=None):
    if sizes is None:
        sizes = {}
    tr = TraceSample(sizes)
    P(tr)
    return tr.trace(varnames)


class TraceSampleLogQ(AbstractTrace):
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
        self.logp = {}

    def sample(self, key, dist, multi_samples=True, plate=None):
        assert key not in self.data
        assert key not in self.samples
        assert key not in self.logp
            
        sample_dims = []
        if plate is not None:
            sample_dims.append(self.plates[plate])
        if multi_samples:
            sample_dims.append(self.Kdim)

        sample = dist.sample(reparam=self.reparam, sample_dims=sample_dims)
        if not multi_samples:
            assert self.Kdim not in sample.dims, "Multiple samples are coming into this variable, so we can't stop it giving multiple samples at the output"

        self.samples[key] = sample
        self.logp[key] = dist.log_prob(sample)
        

class TraceLogP(AbstractTrace):
    def __init__(self, trq):
        self.trq = trq

        self.samples = {}
        self.logp = {}

        self.Kname_to_Kdim = {}
        self.var_to_Kname = {}

    @property
    def data(self):
        return self.trq.data

    def sample(self, key, dist, group=None, plate=None):
        assert key not in self.samples
        assert key not in self.logp
        assert key not in self.var_to_Kname

        if key in self.data:
            sample = self.data[key]
        else:
            Kname = f'K_{key if (group is None) else group}'
            if Kname in self.Kname_to_Kdim:
                Kdim = self.Kname_to_Kdim[Kname]
                assert Kdim.size == self.trq.Kdim.size
            else:
                Kdim = Dim(Kname, self.trq.Kdim.size)
            self.var_to_Kname[key] = Kname

            sample_q = self.trq[key]
            sample = sample_q.order(self.trq.Kdim)[Kdim]
            self.samples[key] = sample

        self.logp[key] = dist.log_prob(sample)
