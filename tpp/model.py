from warnings import warn
import torch.nn as nn
from .traces import TraceQ, TraceP, TracePred
from .Sample import Sample
from .utils import *

class Q(nn.Module):
    """
    A thin wrapper on nn.Module, which is only necessary if we want to
    learn separate parameters for each latent variable in a plate. In that 
    case, the learned parameters need to be torchdim tensors, and this class
    allows that.

    In particular, any parameter with a plate should be registered with 
    ```
    class MyQ(Q):
        def __init__(self):
            super().__init__()
            self.reg_param("a", t.ones(3,4), ("plate_1"))
    ```
    
    The size of this plate will also feed into the inference of plate sizes
    performed in `Model`.
    """
    def __init__(self):
        super().__init__()
        self._plates = {}
        self._params = nn.ParameterDict()
        self._dims = {}

    def reg_param(self, name, tensor, dims=None):
        """ Register a parameter with plate dimensions.
        Args:
            name (str):            the name of the parameter.
            tensor (torch.Tensor): the tensor (may be named).
            dims:                  the dimension names, starting from dimension 0.

        You may specify the plates either by providing a named tensor, or by 
        providing a dims argument, but not both!
        """
        if (dims is not None) and any(name is not None for name in tensor.names):
            raise Exception("Names should be provided either using a named tensor _or_ using the dims optional argument.  Names provided in a named tensor and in the dims optional argument.")

        #Save unnamed parameter
        self._params[name] = nn.Parameter(tensor.rename(None))

        #Put everything into names, and generate names for each dim.
        if dims is not None:
            tensor = tensor.rename(*dims, *((tensor.ndim - len(dims))*[None]))
        self._plates = extend_plates_with_named_tensor(self._plates, tensor)

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
    """Model class.
    Args:
        P:    The generative model, written as a function that takes a trace.
        Q:    The proposal / approximate posterior. Optional. If not provided,
              then we use the prior as Q.
        data: Any non-minibatched data. This is usually used in statistics,
              where we have small-medium data that we can reason about as a 
              block. This is a dictionary mapping variable name to named-tensors
              representing the data. We infer plate sizes from the sizes of 
              the named dimensions in data (and from the sizes of any parameters
              in Q).
    """
    def __init__(self, P, Q=lambda tr: None, data=None):
        super().__init__()
        self.P = P
        self.Q = Q

        if data is None:
            data = {}

        #plate dimensions can come in through:
        #  parameters in Q
        #  non-minibatched data passed to the model.
        #  minibatched data passed to e.g. model.elbo(...)
        #here, we gather plate dimensions from the first two.
        #in _sample, we gather plate dimensions from the last one.
        Q_plates = Q._plates if hasattr(Q, "_plates") else {}
        self.platedims = extend_plates_with_named_tensors(Q_plates, data.values())
        self.data   = named2dim_tensordict(self.platedims, data)

    def _sample(self, K, reparam, data, memory_diagnostics=False):
        """
        Internal method that actually runs P and Q.
        """
        if data is None:
            data = {}
        platedims = extend_plates_with_named_tensors(self.platedims, data.values())
        data = named2dim_tensordict(platedims, data)

        all_data = {**self.data, **data}
        if 0==len(all_data):
            raise Exception("No data provided either to the model or to the called method")
        assert len(all_data) == len(self.data) + len(data)

        #sample from approximate posterior
        trq = TraceQ(K, all_data, platedims, reparam)
        self.Q(trq)
        #compute logP
        trp = TraceP(trq, memory_diagnostics=memory_diagnostics)
        self.P(trp)

        return Sample(trp)

    def elbo(self, K, data=None, reparam=True):
        """Compute the ELBO.
        Args:
            K:       the number of samples drawn for each latent variable.
            data:    Any minibatched data.
            reparam: Whether to use the reparameterisation trick.  If you want to use the
                     ELBO as an objective in VI, then this needs to be True (and it is 
                     true by default).  However, sampling with reparam=True will fail if 
                     you have discrete latent variables. Indeed, you can't do standard VI
                     with discrete latents. That said, if you have discrete latent
                     variables, you may still want to compute a bound on the model
                     evidence, and that's probably the only case where reparam=False makes
                     sense.
        """
        if not reparam:
            warn("Evaluating the ELBO without reparameterising.  This can be valid, e.g. if you're just trying to compute a bound on the model evidence.  But it won't work if you try to train the generative model / approximate posterior using the non-reparameterised ELBO as the objective.")
        return self._sample(K, reparam, data).elbo()

    def rws(self, K, data=None):
        """Compute RWS objectives
        Args:
            K:       the number of samples drawn for each latent variable.
            data:    Any minibatched data.
        Returns:
            p_obj: Objective for the P update
            q_obj: Objective for the wake-phase Q update

        RWS ...
        """
        return self._sample(K, False, data).rws()

    def weights(self, K, data=None):
        """Compute marginal importance weights
        Args:
            K:       the number of samples drawn for each latent variable.
            data:    Any minibatched data.
        Returns:
            A dictionary mapping the variable name to a tuple of weights and samples.
            These weights and samples may be used directly, or may be processed to
            give e.g. moments, ESS etc. using the functions in tpp.postproc
        """
        return self._sample(K, False, data).weights()

    def importance_samples(self, K, N, data=None):
        """Compute posterior samples
        Args:
            K:       the number of samples drawn for each latent variable.
            N:       the number of importance samples returned.
            data:    Any minibatched data.
        Returns:
            A dictionary mapping the variable name to the posterior sample.

        Notes:
            * This is only really useful for prediction. If you're looking 
              for moments, you should use importance weights processed by 
              tpp.postproc.  This will be more accurate...
        """
        N = Dim('N', N)
        return self._sample(K, False, data)._importance_samples(N)

    def _predictive(self, K, N, data_all=None, platesizes_all=None):
        sample = self._sample(K, False, None)

        if (data_all is not None):
            if not any(sample.trp.data[dataname].numel() < dat.numel() for (dataname, dat) in data_all.items()):
                raise Exception(f"None of the tensors provided data_all is bigger than those provided at training time, so it doesn't make sense to make predictions.  If you just want posterior samples, use model.importance_samples(...)")
        if (platesizes_all is not None):
            if not any(self.trp.trq.plates[platename] < size for (platename, size) in sizes_all.items()):
                raise Exception("None of the sizes provided in sizes_all are bigger than those in the training data, so we can't make any predictions.  If you just want posterior samples, use model.importance_samples")

        N = Dim('N', N)
        post_samples = sample._importance_samples(N)
        tr = TracePred(N, post_samples, sample.trp.data, data_all, sample.trp.platedims, platesizes_all)
        self.P(tr)
        return tr, N

    def predictive_samples(self, K, N, platesizes_all=None):
        if platesizes_all is None:
            platesizes_all = {}
        trace_pred, N = self._predictive(K, N, None, platesizes_all)
        #Convert everything to named
        #Return a dict mapping
        #Convert everything to named
        return trace_pred.samples_all

    def predictive_ll(self, K, N, data_all):
        """
        Run as (e.g. for plated_linear_gaussian.py)

        >>> obs = t.randn((4, 6, 8), names=("plate_1", "plate_2", "plate_3"))
        >>> model.predictive_ll(5, 10, data_all={"obs": obs})
        """

        trace_pred, N = self._predictive(K, N, data_all, None)
        lls_all   = trace_pred.ll_all
        lls_train = trace_pred.ll_train
        assert set(lls_all.keys()) == set(lls_train.keys())

        result = {}
        for varname in lls_all:
            ll_all   = lls_all[varname]
            ll_train = lls_train[varname]

            #print(varname)

            dims_all   = [dim for dim in ll_all.dims   if dim is not N]
            dims_train = [dim for dim in ll_train.dims if dim is not N]
            assert len(dims_all) == len(dims_train)

            #print(dims_all)
            #print(dims_train)
            if 0 < len(dims_all):
                ll_all   = ll_all.sum(dims_all)
                ll_train = ll_train.sum(dims_train)
            #print(ll_all)
            #print(ll_train)
            result[varname] = (ll_all - ll_train).mean(N)

        return result
