import torch as t
import torch.nn as nn
from .dist import *
from .utils import *
from .alan_module import AlanModule
from .exp_fam_mixin import *


class AMMP_IS(AlanModule):
    """
    Adaptive Multiple Massively Parallel Importance Sampling
    """
    def __init__(self, platesizes=None, sample_shape=()):
        super().__init__()



        init_conv = self.default_init_conv
        init_means = self.conv2mean(**init_conv)

        if platesizes is None:
            platesizes = {}
        shape = [*platesizes.values(), *sample_shape]
        names = [*platesizes.keys(), *(len(sample_shape) * [None])]

        self.meannames = tuple(f'mean_{i}' for i in range(len(self.sufficient_stats)))
        for (meanname, init_mean) in zip(self.meannames, init_means):
            self.register_buffer(meanname, t.full(shape, init_mean, dtype=t.float64).rename(*names))

        self.avgmeannames = tuple(f'mean_avg_{i}' for i in range(len(self.sufficient_stats)))
        for (meanname, init_mean) in zip(self.avgmeannames, init_means):
            self.register_buffer(meanname, t.full(shape, 0, dtype=t.float64).rename(*names))

        self.natnames = tuple(f'nat_{i}' for i in range(len(self.sufficient_stats)))
        for natname in self.natnames:
            self.register_parameter(natname, nn.Parameter(t.zeros(shape, dtype=t.float64).rename(*names)))
        self.reset_nats()

        self.platenames = tuple(platesizes.keys())

        self.Ms = [[] for i in range(len(self.sufficient_stats))]


    @property
    def dim_means(self):
        return [getattr(self, meanname) for meanname in self.meannames]
    @property
    def named_means(self):
        return [self.get_named_tensor(meanname) for meanname in self.meannames]
    @property
    def named_avgmeans(self):
        return [self.get_named_tensor(meanname) for meanname in self.avgmeannames]
    @property
    def dim_nats(self):
        return [getattr(self, natname) for natname in self.natnames]
    @property
    def named_nats(self):
        return [self.get_named_tensor(natname) for natname in self.natnames]
    @property
    def named_grads(self):
        return [self.get_named_grad(natname) for natname in self.natnames]

    def _update(self, sample, lr, explts):
        m_one_iter = self.sample2mean(sample.weights(), self.index)
        with t.no_grad():
            idx = 0
            
            for idx, (mean, avgmean,m_one)  in enumerate(zip(self.named_means, self.named_avgmeans, m_one_iter)):
                self.Ms[idx].append(m_one)
                # avgmean.data.copy_(eta * m_one + (1 - eta) * avgmean)

                avgmean.data.copy_(sum([explt * m for explt, m in zip(explts, self.Ms[idx])]))
                mean.data.copy_(lr*avgmean + (1-lr)*mean) 
                idx += 1

        self.reset_nats()

    def reset_nats(self):
        with t.no_grad():
            new_nats = self.mean2nat(*self.named_means)
            for old_nat, new_nat in zip(self.named_nats, new_nats):
                old_nat.data.copy_(new_nat)

    def __call__(self):
        return self.dist(**self.nat2conv(*self.dim_nats))

    def local_parameters(self):
        return []

class AMMP_ISNormal(AMMP_IS, NormalMixin):
    pass
class AMMP_ISMvNormal(AMMP_IS, MvNormalMixin):
    pass
class AMMP_ISBernoulliLogits(AMMP_IS, BernoulliLogitsMixin):
    pass
class AMMP_ISPoisson(AMMP_IS, PoissonMixin):
    pass
class AMMP_ISExponential(AMMP_IS, ExponentialMixin):
    pass
class AMMP_ISDirichlet(AMMP_IS, DirichletMixin):
    pass
class AMMP_ISBeta(AMMP_IS, BetaMixin):
    pass
class AMMP_ISGamma(AMMP_IS, GammaMixin):
    pass
