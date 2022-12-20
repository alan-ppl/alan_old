import torch.nn as nn
from .traces import TraceQ, TraceP, TracePred
from .Sample import Sample
from .utils import *

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
        plates = Q._plates if hasattr(Q, "_plates") else {}
        self.data, self.plates = named2dim_data(data, plates)

    def sample(self, K, reparam, data, memory_diagnostics=False):
        data, plates = named2dim_data(data, self.plates)
        all_data = {**self.data, **data}
        assert len(all_data) == len(self.data) + len(data)

        #sample from approximate posterior
        trq = TraceQ(K, all_data, plates, reparam)
        self.Q(trq)
        #compute logP
        trp = TraceP(trq, memory_diagnostics=memory_diagnostics)
        self.P(trp)

        return Sample(trp)

    def elbo(self, K, data=None):
        return self.sample(K, True, data).elbo()

    def rws(self, K, data=None):
        return self.sample(K, False, data).rws()

    def weights(self, K, data=None):
        return self.sample(K, False, data).weights()

    def importance_samples(self, K, N, data=None):
        N = Dim('N', N)
        return self.sample(K, False, data)._importance_samples(N)

    def predictive(self, K, N, data_train=None, data_all=None, sizes_all=None):
        sample = self.sample(K, False, data_train)
        N = Dim('N', N)
        post_samples = sample._importance_samples(N)
        tr = TracePred(N, post_samples, sample.trp.data, sample.trp.trq.plates, data_all=data_all, sizes_all=sizes_all)
        self.P(tr)
        return tr, N

    def predictive_samples(self, K, N, data_train=None, sizes_all=None):
        trace_pred, N = self.predictive(K, N, data_train=data_train, sizes_all=sizes_all)
        #Convert everything to named
        #Return a dict mapping
        #Convert everything to named
        return trace_pred.samples_all

    def predictive_ll(self, K, N, data_train=None, data_all=None):
        """
        Run as (e.g. for plated_linear_gaussian.py)

        >>> obs = t.randn((4, 6, 8), names=("plate_1", "plate_2", "plate_3"))
        >>> model.predictive_ll(5, 10, data_all={"obs": obs})
        """
        trace_pred, N = self.predictive(K, N, data_train=data_train, data_all=data_all)
        lls_all   = trace_pred.ll_all
        lls_train = trace_pred.ll_train
        assert set(lls_all.keys()) == set(lls_train.keys())

        result = {}
        for varname in lls_all:
            ll_all   = lls_all[varname]
            ll_train = lls_train[varname]

            print(varname)

            dims_all   = [dim for dim in ll_all.dims   if dim is not N]
            dims_train = [dim for dim in ll_train.dims if dim is not N]
            assert len(dims_all) == len(dims_train)

            print(dims_all)
            print(dims_train)
            if 0 < len(dims_all):
                ll_all   = ll_all.sum(dims_all)
                ll_train = ll_train.sum(dims_train)
            print(ll_all)
            print(ll_train)
            result[varname] = (ll_all - ll_train).mean(N)

        return result
