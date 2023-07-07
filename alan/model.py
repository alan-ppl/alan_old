from warnings import warn
import torch.nn as nn
from . import traces
from .Sample import Sample, SampleGlobal
from .utils import *
from torch.nn.functional import threshold

from .alan_module import AlanModule

from torch.nn.functional import softplus
class SampleMixin():
    r"""
    A mixin for :class:`Model` and :class:`ConditionedModel` that introduces the sample_... methods
    Requires methods:
        self.P(tr, ...)
        self.Q(tr, ...)
        self.check_device(device)
    """
    def dims_data_inputs(self, data, inputs, platesizes, device, use_model=True):
        r"""
        Adds the right dimensions to *data* and *inputs*.

        Args:
            data (Dict): **Dict** containing data
            inputs (Dict): **Dict** containing inputs (covariates)
            platesizes (Dict): **Dict** mapping from dim name to size
            device (torch.device): Device to put data and inputs on
            use_model (Bool): **True** to use the model to determine dims
        """
        #check model and/or self.data + self.inputs on ConditionModel are on desired device
        self.check_device(device)
        #deal with possibility of None defaults
        data       = none_empty_dict(data)
        inputs     = none_empty_dict(inputs)
        platesizes = none_empty_dict(platesizes)

        #place on device
        data   = {k: v.to(device=device, dtype=t.float64) for (k, v) in data.items()}
        inputs = {k: v.to(device=device, dtype=t.float64) for (k, v) in inputs.items()}
        

        platedims = self.platedims if use_model else {}
        platedims = extend_plates_with_named_tensors(platedims, [*data.values(), *inputs.values()])
        platedims = extend_plates_with_sizes(platedims, platesizes)

        if hasattr(self, "data") and use_model:
            assert 0 == len(set(self.data).intersection(data))
            data = {**self.data, **data}

        if hasattr(self, "inputs") and use_model:
            assert 0 == len(set(self.inputs).intersection(inputs))
            inputs = {**self.inputs, **inputs}

        data   = named2dim_tensordict(platedims, data)
        inputs = named2dim_tensordict(platedims, inputs)
        return platedims, data, inputs

    def sample_same(self, *args, **kwargs):
        r"""
        Sample from the model where each of the K particles is sampled conditioned on the K'th parent
        """
        return self.sample_tensor(traces.TraceQSame, *args, **kwargs)

    def sample_cat(self, *args, **kwargs):
        r"""
        Sample from the model where each of the K particles is sampled conditioned on a parent selected using
        a categorical distribution
        """
        return self.sample_tensor(traces.TraceQCategorical, *args, **kwargs)

    def sample_perm(self, *args, **kwargs):
        r"""
        Sample from the model where each of the K particles is sampled conditioned on a parent selected using a permutation
        """
        return self.sample_tensor(traces.TraceQPermutation, *args, **kwargs)

    def sample_global(self, *args, **kwargs):
        r"""
        Sample from the model where K samples are drawn from the whole latent space with no combinations
        """
        return self.sample_tensor(traces.TraceQGlobal, *args, **kwargs)

    def sample_tensor(self, trace_type, K, reparam=True, data=None, inputs=None, platesizes=None, device=t.device('cpu'), lp_dtype=t.float64, lp_device=None):
        r"""
        Internal method that actually runs *P* and *Q*.

        Args:
            trace_type: One of:
                            - traces.TraceQSame
                            - traces.TraceQCategorical
                            - traces.TraceQPermutation
                            - traces.TraceQGlobal
            K (int): Number of K samples
            reparam (bool): **True** to sample using reparameterisation trick (Not available for all dists)
            data (Dict): **Dict** containing data
            inputs (Dict): **Dict** containing inputs (covariates)
            platesizes (Dict): **Dict** mapping from dim name to size
            device (torch.device): Device to put data and inputs on

        Returns:
            Sample (:class:`alan.Sample.Sample`): Sample object
        """
        platedims, data, inputs = self.dims_data_inputs(data, inputs, platesizes, device)

        #if 0==len(all_data):
        #    raise Exception("No data provided either to the Model(...) or to e.g. model.elbo(...)")
        #for dataname in self.data:
        #    if dataname in data:
        #        raise Exception(f"Data named '{dataname}' were provided to Model(...) and e.g. model.elbo(...).  You should provide data only once.  You should usually provide data to Model(...), unless you're minibatching, in which case it needs to be provided to e.g. model.elbo(...)")
        #if 0 != len(self.data) and 0 != len(data):
        #    warn("You have provided data to Model(...) and e.g. model.elbo(...). There are legitimate uses for this, but they are very, _very_ unusual.  You should usually provide all data to Model(...), unless you're minibatching, in which case that data needs to be provided to e.g. model.elbo(...).  You may have some minibatched and some non-minibatched data, but very likely you don't.")

        #sample from approximate posterior
        trq = trace_type(K, data, platedims, reparam, device, lp_dtype)
        self.Q(trq, **inputs)
        #compute logP
        trp = traces.TraceP(trq)
        self.P(trp, **inputs)

        return Sample(trp, lp_dtype=lp_dtype, lp_device=lp_device)

    def sample_prior(self, N=None, reparam=True, inputs=None, platesizes=None, device=t.device('cpu'), varnames=None):
        """Draw samples from a generative model (with no data).

        Args:
            N (int):        The number of samples to draw
            reparam (bool): **True** to sample using reparameterisation trick (Not available for all dists)
            inputs (Dict): **Dict** containing inputs (covariates)
            platesizes (Dict): **Dict** mapping from dim name to size
            device (torch.device): Device to put data and inputs on
            varnames (iterable): An iterable of the variables to return

        Returns:
            A dictionary mapping from variable name to sampled value,
            represented as a named tensor (e.g. so that it is suitable
            for use as data).
        """
        platedims, data, inputs = self.dims_data_inputs({}, inputs, platesizes, device)

        with t.no_grad():
            tr = traces.TraceSample(N, platedims, device)
            self.P(tr, **inputs)

        if isinstance(varnames, str):
            varnames = (varnames,)
        elif varnames is None:
            varnames = tr.samples.keys()

        return {varname: dim2named_tensor(tr.samples[varname]) for varname in varnames}

    def _predictive(self, sample, N, data_all=None, inputs_all=None, platesizes_all=None):
        assert isinstance(sample, (Sample, SampleGlobal))
        assert isinstance(N, int)
        N = Dim('N', N)
        #platedims, data, inputs = self.dims_data_inputs(data_all, covariates_all, platesizes_all, device)
        post_samples = sample._importance_samples(N)

        platedims_all, data_all, inputs_all = self.dims_data_inputs(data_all, inputs_all, platesizes_all, device=sample.device, use_model=False)

        tr = traces.TracePred(
            N, post_samples,
            sample.trp.data, data_all,
            sample.trp.platedims, platedims_all,
            device=sample.device
        )
        self.P(tr, **inputs_all)
        return tr, N

    def predictive_samples(self, sample, N, inputs_all=None, platesizes_all=None):
        if platesizes_all is None:
            platesizes_all = {}
        trace_pred, N = self._predictive(sample, N, data_all=None, inputs_all=inputs_all, platesizes_all=platesizes_all)
        #Convert everything to named
        #Return a dict mapping
        #Convert everything to named
        return trace_pred.samples

    def predictive_ll(self, sample, N, data_all, inputs_all=None):
        """
        Run as (e.g. for plated_linear_gaussian.py)

        >>> obs = t.randn((4, 6, 8), names=("plate_1", "plate_2", "plate_3"))
        >>> model.predictive_ll(5, 10, data_all={"obs": obs})
        """

        trace_pred, N = self._predictive(sample, N, data_all, inputs_all, None)
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
            #result[varname] = (ll_all - ll_train).mean(N)
            result[varname] = logmeanexp_dims(ll_all - ll_train, (N,))

        return result

    def update(self, lr, sample):
        """
        Will call update on Model
        """
        assert not sample.reparam
        _, q_obj = sample.rws()
        (q_obj).backward()

        model = self.model if isinstance(self, ConditionedModel) else self
        for mod in model.modules():
            if hasattr(mod, '_update'):
                mod._update(lr)
        self.zero_grad()

    def ammpis_update(self, lr, sample):
        """
        Will call update on Model
        """
        # assert not sample.reparam


        HQ_t = getattr(self.model, 'HQ_t')
        HQ_t_minus_1 = getattr(self.model, 'HQ_t_minus_1')
        model = self.model if isinstance(self, ConditionedModel) else self


        
        hq = 0
        for mod in model.modules():
            if hasattr(mod, '_update'):
                sig = mod.mean2conv(*mod.named_means)['scale'].sum()
                hq += 1/2 * t.log(2*t.pi*sig) + 1/2

        HQ_t.data.copy_(hq) 
        self.model.HQs.append(HQ_t.item())
        #Trying using a downweighting term so our early broad posterior's high importance weight variance
        #doesn't cause us to have overly narrow t+1 posteriors

        # simplest method
        # dt = HQ_t_minus_1 - HQ_t + 0.1
        
        # Using relu
        # dt = t.nn.functional.relu(HQ_t_minus_1 - HQ_t)
        # print(dt)

        # l_tot = getattr(self.model, 'l_tot')
        # l_one_iter = sample.elbo().item()
        # l_tot.data.add_(softplus(l_one_iter + dt - l_tot) - dt)
        # eta = t.exp(l_one_iter - l_tot)
        

        weights = [t.exp(-t.nn.functional.relu(hq - HQ_t)) for hq in self.model.HQs]
        l_tot = getattr(self.model, 'l_tot')
        l_one_iter = sample.elbo()
        l_tot.data.copy_(t.log(sum([w*p for w, p in zip(weights, self.model.P_one_iters)])))
        print(self.model.P_one_iters)
        self.model.P_one_iters.append(t.exp(l_one_iter))


        for mod in model.modules():
            if hasattr(mod, '_update'):
                mod._update(sample, weights, lr, t.exp(l_tot), self.model.P_one_iters)

        HQ_t_minus_1.data.copy_(HQ_t.data)
        self.zero_grad()

    def ng_update(self, lr, sample):
        assert sample.reparam
        sample.elbo().backward()

        model = self.model if isinstance(self, ConditionedModel) else self
        for mod in model.modules():
            if hasattr(mod, '_ng_update'):
                mod._ng_update(lr)
        self.zero_grad()

    def local_parameters(self):
         return super().parameters(recurse=False)

    def grad_parameters(self):
        #Avoids returning Js so they don't get passed into an optimizer.
        #This can cause problems if other methods (e.g. zero_grad) use
        #self.parameters (e.g. see ml_update).
        result = local_parameters()
        for mod in self.children():
            params = mod.grad_parameters() if isinstance(mod, Model) else mod.parameters()
            for param in params:
                result.add(param)
        return list(result)

class NestedModel():
    """
    Returned from model.forward(*args, **kwargs)
    Wraps up a model and its inputs for nested use.
    """
    def __init__(self, model, args, kwargs):
        self.model = model
        self.args = args
        self.kwargs = kwargs

    def P(self, tr):
        self.model.P(tr, *self.args, **self.kwargs)
    def Q(self, tr):
        self.model.Q(tr, *self.args, **self.kwargs)

def none_empty_dict(x):
    return {} if x is None else x

class ConditionedModel(SampleMixin):
    """
    NOT a nn.Module
    Represents a model bound to model to data and inputs.
    Returned
    bound_model = model.bind(data=..., inputs=...)
    bound_model.sample_mp(K) == model.sample_mp(K, data=..., inputs=...)
    """
    def __init__(self, model, data, inputs, platesizes):
        super().__init__()
        self.model  = model
        self.data   = none_empty_dict(data)
        self.inputs = none_empty_dict(inputs)
        platesizes  = none_empty_dict(platesizes)

        self.platedims = extend_plates_with_sizes(model.platedims, platesizes)
        tensors = [*self.data.values(), *self.inputs.values()]
        self.platedims = extend_plates_with_named_tensors(self.platedims, tensors)

    def P(self, tr, *args, **kwargs):
        self.model.P(tr, *args, **kwargs)

    def Q(self, tr, *args, **kwargs):
        self.model.Q(tr, *args, **kwargs)

    def parameters(self):
        return self.model.parameters()

    def zero_grad(self):
        return self.model.zero_grad()

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        self.data   = {k: v.to(*args, **kwargs) for (k, v) in self.data.items()}
        self.inputs = {k: v.to(*args, **kwargs) for (k, v) in self.inputs.items()}

    def check_device(self, device):
        self.model.check_device(device)

        device = t.device(device)
        for x in [*self.data.values(), *self.inputs.values()]:
            assert x.device == device

class Model(SampleMixin, AlanModule):
    """Model class.
    A Model must be provided with a generative model, P, and an approximate
    posterior / proposal, Q.  There are two options.  They can be provided
    as arguments, or defined in a subclass.

    To provide P and Q as an argument, use `Model(P, Q)`, or `Model(P)` (in
    which case P is used as Q).  P and Q must be callable, in the form
    `P(tr, ...)` or `Q(tr, ...`, where ... is any extra inputs (and is the
    same for P and Q.

    Alternatively, you can subclass model, overriding __init__, and providing
    subclass.P(tr, ...) and subclass.Q(tr, ...) as methods.
    """
    def __init__(self, P=None, Q=None):
        super().__init__()
        if P is not None:
            assert not hasattr(self, 'P')
            self.P = P
        assert hasattr(self, 'P')

        if Q is not None:
            assert not hasattr(self, 'Q')
            self.Q = Q

        #Default to using P as Q if Q is not defined.
        if not hasattr(self, 'Q'):
            self.Q = P
        
        assert not hasattr(self, 'l_tot')
        self.register_buffer('l_tot', t.tensor(-1e15, dtype=t.float64))
        self.register_buffer('HQ_t', t.tensor(0.0, dtype=t.float64))
        self.register_buffer('HQ_t_minus_1', t.tensor(0.0, dtype=t.float64))
        self.HQs = [t.tensor(0.0, dtype=t.float64)]
        self.P_one_iters = [t.tensor(1, dtype=t.float64)]

    def forward(self, *args, **kwargs):
        return NestedModel(self, args, kwargs)

    def condition(self, data=None, inputs=None, platesizes=None):
        r"""
        Args:
            data:   Any non-minibatched data. This is usually used in statistics,
                    where we have small-medium data that we can reason about as a
                    block. This is a dictionary mapping variable name to named-tensors
                    representing the data. We infer plate sizes from the sizes of
                    the named dimensions in data (and from the sizes of any parameters
                    in Q).
            inputs: Any non-minibatched data. This is usually used in statistics,
                    where we have small-medium data that we can reason about as a
                    block. This is a dictionary mapping variable name to named-tensors
                    representing the data. We infer plate sizes from the sizes of
                    the named dimensions in data (and from the sizes of any parameters
                    in Q).
        """
        return ConditionedModel(self, data, inputs, platesizes)

    def check_device(self, device):
        device = t.device(device)
        for x in [*self.parameters(), *self.buffers()]:
            if x.device != device:
                x.to(device)
