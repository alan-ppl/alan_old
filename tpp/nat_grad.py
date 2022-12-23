import torch as t
import torch.nn as nn
from .model import Q
from .exp_fam_conversions import conv_dict

class NatParam():
    def __init__(self, dist, sample):
        self.conv = conv_dict[dist]
        means = tuple(f(sample) for f in self.conv.sufficient_stats)
        self.nats = conv.mean2nat(*means)

    def dist(self):
        return self.conv.dist(*self.conv.nat2conv(*self.nats))

    def update(self, lr):
        with torch.no_grad():
            old_nats  = self.nats
            old_means = self.conv.nat2mean(*old_nats) #This is the efficient direction.
            new_means = [mean + lr * nat.grad for (nat, mean) in zip(old_nats, old_means)]
            new_nats  = self.conv.mean2nat(*old_nats)
            self.nats = [nat.detach().requires_grad_() for nat in new_nats]
            for nat in self.nats:
                nat.retain_grad()
            

class NatQ():
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

        #Remove any samples in the data.
        samples = {key: value for (key, value) in samples.items() if key not in data}

        for (key, dist) in dists.items():
            if key not in samples:
                raise Exception("Variable '{key}' not present in sample")

        keys_in_samples_but_not_dists = set(samples.keys()).difference(dists.keys())
        if 0!=len(keys_in_samples_but_not_dists):
            raise Exception("Some variables ({keys_in_samples_but_not_dists}) were given in samples but not dists.  Most likely, these variables will be sampled from the prior.")

        self.natparams = {key: NatParam(dists[key], samples[key]) for key in dists}
        
    def forward(self, tr):
        for key, natparam in self.natparams.items():
            tr[key] = sample

    def update(self, lr):
        for natparam in self.natparams.values():
            natparam.update(lr)

  
