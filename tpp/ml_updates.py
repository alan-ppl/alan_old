import torch as t
import torch.nn as nn
from .model import QModule
from .exp_fam_conversions import conv_dict
from .postproc import mean

class MLParam(QModule):
    def __init__(self, dist, sample):
        super().__init__()
        self.conv = conv_dict[dist]

        self.varnames = tuple(f'm{i}' for i in range(len(self.conv.sufficient_stats)))
        for (varname, f) in zip(self.varnames, self.conv.sufficient_stats):
            self.register_buffer(varname, mean(f(sample)))

    @property
    def dim_means(self):
        return [getattr(self, varname) for varname in self.varnames]

    @property
    def named_means(self):
        return [self.get_named_tensor(varname) for varname in self.varnames]

    def dist(self):
        return self.conv.dist(*self.conv.mean2conv(*self.dim_means))

    def update(self, lr, sample):
        with t.no_grad():
            for (m, f) in zip(self.named_means, self.conv.sufficient_stats):
                m.data.mul_(1-lr).add_(mean(f(sample)).align_as(m), alpha=lr)

class MLQ(QModule):
    def __init__(self, samples, dists, data={}):
        """
        Args:
            sample: dict mapping variable name to a prior sample, to obtain the prior moments.
            dist:   dict mapping variable name to the distribution (as the corresponding tpp object, e.g. tpp.Normal).
            data:   dict mapping data names to data tensors or iterable just containing data names, to avoid making an approximate posterior for the data.

        Note: not a nn.Module, so that .parameters() doesn't give the
        natural parameters to the optimizer.  That said, the parameters
        do get gradients.  
        """
        super().__init__()

        #Remove any samples in the data.
        samples = {key: value for (key, value) in samples.items() if key not in data}

        for (key, dist) in dists.items():
            if key not in samples:
                raise Exception("Variable '{key}' not present in sample")

        keys_in_samples_but_not_dists = set(samples.keys()).difference(dists.keys())
        if 0!=len(keys_in_samples_but_not_dists):
            raise Exception("Some variables ({keys_in_samples_but_not_dists}) were given in samples but not dists.  Most likely, these variables will be sampled from the prior.")

        self.natparams = nn.ModuleDict({key: MLParam(dists[key], samples[key]) for key in dists})
        
    def __call__(self, tr):
        for key, natparam in self.natparams.items():
            tr.sample(key, natparam.dist())

    def update(self, lr, samples):
        for (varname, natparam) in self.natparams.items():
            natparam.update(lr, samples[varname])
