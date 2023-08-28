import torch as t
import torch.nn as nn
from .dist import *
from .utils import *
from .alan_module import AlanModule
from .exp_fam_mixin import *

class ML(AlanModule):
    """
    Isn't quite ML...
    In particular, the RWS wake-phase Q update allows us to in effect compute,
    E_P[log Q]
    If we take Q to be exponential family,
    log Q = eta * T(x) - A(eta)
    Then,
    grad_eta E_P[log Q] = grad_eta [eta * m_0] - grad_eta A(eta)
                        = m_0 - m

                        
                    m_t = m_t-1 + lambda * (-m_t-1 + m)
    So, we can compute the gradient of the log partition function, and then
    """
    def __init__(self, platesizes=None, sample_shape=(), init_conv=None):
        super().__init__()
        if init_conv is None:
            init_conv = self.default_init_conv
        init_means = self.conv2mean(**init_conv)

        if platesizes is None:
            platesizes = {}
        shape = [*platesizes.values(), *sample_shape]
        names = [*platesizes.keys(), *(len(sample_shape) * [None])]

        self.meannames = tuple(f'mean_{i}' for i in range(len(self.sufficient_stats)))
        for (meanname, init_mean) in zip(self.meannames, init_means):
            self.register_buffer(meanname, t.full(shape, init_mean, dtype=t.float64).rename(*names))

        self.natnames = tuple(f'nat_{i}' for i in range(len(self.sufficient_stats)))
        for natname in self.natnames:
            self.register_parameter(natname, nn.Parameter(t.zeros(shape, dtype=t.float64).rename(*names)))
        self.reset_nats()

        self.platenames = tuple(platesizes.keys())

    @property
    def dim_means(self):
        return [getattr(self, meanname) for meanname in self.meannames]
    @property
    def named_means(self):
        return [self.get_named_tensor(meanname) for meanname in self.meannames]

    @property
    def dim_nats(self):
        return [getattr(self, natname) for natname in self.natnames]
    @property
    def named_nats(self):
        return [self.get_named_tensor(natname) for natname in self.natnames]
    @property
    def named_grads(self):
        return [self.get_named_grad(natname) for natname in self.natnames]

    def _update(self, lr):
        with t.no_grad():
            for (mean, grad) in zip(self.named_means, self.named_grads):
                mean.data.add_(grad, alpha=lr)
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

class MLNormal(ML, NormalMixin):
    pass
class MLMvNormal(ML, MvNormalMixin):
    pass
class MLBernoulliLogits(ML, BernoulliLogitsMixin):
    pass
class MLPoisson(ML, PoissonMixin):
    pass
class MLExponential(ML, ExponentialMixin):
    pass
class MLDirichlet(ML, DirichletMixin):
    pass
class MLBeta(ML, BetaMixin):
    pass
class MLGamma(ML, GammaMixin):
    pass
