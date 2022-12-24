import torch as t
import torch.nn as nn
from .model import QModule
from .exp_fam_conversions import conv_dict
from .postproc import mean

class NatParam(QModule):
    def __init__(self, dist, sample):
        super().__init__()
        self.conv = conv_dict[dist]
        means = tuple(mean(f(sample)) for f in self.conv.sufficient_stats)
        nats = self.conv.mean2nat(*means)

        self.varnames = tuple(f'n{i}' for i in range(len(means)))
        for (varname, nat) in zip(self.varnames, nats):
            self.register_parameter(varname, nn.Parameter(nat))

    @property
    def dim_nats(self):
        return [getattr(self, varname) for varname in self.varnames]

    @property
    def named_nats(self):
        return [self.get_named_tensor(varname) for varname in self.varnames]

    def dist(self):
        return self.conv.dist(*self.conv.nat2conv(*self.dim_nats))

    def update(self, lr):
        with t.no_grad():
            old_nats  = self.named_nats
            old_means = self.conv.nat2mean(*old_nats) #This is the efficient direction.
            new_means = [mean + lr * nat.grad for (nat, mean) in zip(old_nats, old_means)]
            new_nats  = self.conv.mean2nat(*new_means)
            for (varname, new_nat) in zip(self.varnames, new_nats):
                nat = self.get_named_tensor(varname)
                nat.data.copy_(new_nat)
            

class NatQ(QModule):
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

        self.natparams = nn.ModuleDict({key: NatParam(dists[key], samples[key]) for key in dists})
        
    def __call__(self, tr):
        for key, natparam in self.natparams.items():
            tr.sample(key, natparam.dist())

    def update(self, lr):
        for natparam in self.natparams.values():
            natparam.update(lr)

    def zero_grad(self):
        for natparam in self.natparams.values():
            natparam.zero_grad()

  
