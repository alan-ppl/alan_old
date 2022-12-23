from warnings import warn
import torch.nn as nn
from .traces import TraceQ, TraceP, TracePred
from .Sample import Sample
from .utils import *

class QModule(nn.Module):
    def get_named_tensor(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        return None

    def __getattr__(self, name):
        tensor = self.get_named_tensor(name)
        if tensor is not None:
            if not hasattr(self, "_platedims"):
                raise Exception("Cannot return parameter or buffer, as self._platedims is not set.  This happens when Q is given to Model")
            return named2dim_tensor(self._platedims, tensor)
        else:
            return super().__getattr__(name)

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
        self.platedims = extend_plates_with_named_tensors({}, list(Q.parameters()) + list(Q.buffers()))
        for mod in Q.modules():
            if isinstance(mod, QModule):
                assert not hasattr(mod, "_platedims")
                mod._platedims = self.platedims
            else:
                for x in list(mod.parameters(recurse=False)) + list(mod.buffers(recurse=False)):
                    if any(name is not None in x.names):
                        raise Exception("Named parameter on an nn.Module.  To specify plates in approximate posteriors correctly, we need to use QModule in place of nn.Module")

        self.platedims = extend_plates_with_named_tensors(self.platedims, data.values())
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
            raise Exception("No data provided either to the Model(...) or to e.g. model.elbo(...)")
        for dataname in self.data:
            if dataname in data:
                raise Exception(f"Data named '{dataname}' were provided to Model(...) and e.g. model.elbo(...).  You should provide data only once.  You should usually provide data to Model(...), unless you're minibatching, in which case it needs to be provided to e.g. model.elbo(...)")
        assert len(all_data) == len(self.data) + len(data)
        if 0 != len(self.data) and 0 != len(data):
            warn("You have provided data to Model(...) and e.g. model.elbo(...). There are legitimate uses for this, but they are very, _very_ unusual.  You should usually provide all data to Model(...), unless you're minibatching, in which case that data needs to be provided to e.g. model.elbo(...).  You may have some minibatched and some non-minibatched data, but very likely you don't.")

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
