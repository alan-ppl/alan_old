import torch as t
import torch.nn as nn
from .dist import *
from .utils import *
from .qmodule import QModule
from .exp_fam_mixin import *

class Tilted(QModule):
    """
    Isn't quite NG...
    In particular, the RWS wake-phase Q update allows us to in effect compute,
    E_P[log Q]
    If we take Q to be exponential family,
    log Q = eta * T(x) - A(eta)
    Then,
    grad_eta E_P[log Q] = grad_eta [eta * m_0] - grad_eta A(eta)
                        = m_0 - m
    """
    def __init__(self, platesizes=None, sample_shape=()):
        super().__init__()

        if platesizes is None:
            platesizes = {}
        shape = [*platesizes.values(), *sample_shape]
        names = [*platesizes.keys(), *(len(sample_shape) * [None])]

        self.natnames = tuple(f'nat_{i}' for i in range(len(self.sufficient_stats)))
        for natname in self.natnames:
            self.register_parameter(natname, nn.Parameter(t.zeros(shape).rename(*names)))

        self.platenames = tuple(platesizes.keys())

    @property
    def dim_nats(self):
        return [getattr(self, natname) for natname in self.natnames]
    @property
    def named_nats(self):
        return [self.get_named_tensor(natname) for natname in self.natnames]

    def forward(self, prior):
        if not isinstance(prior, self.dist):
            raise(f"{type(self)} can only be combined with {type(self.dist)} distributions")
        prior_convs = self.canonical_conv(**prior.dim_args)
        prior_nats = self.conv2nat(**prior_convs)
        post_nats = tuple(prior+post for (prior, post) in zip(prior_nats, self.dim_nats))
        return self.dist(**self.nat2conv(*post_nats))

    def update(self, lr):
        with t.no_grad():
            #Initialize means at old values
            means = self.nat2mean(*self.named_nats)
            #Update means to new values
            for (mean, nat) in zip(means, self.named_nats):
                mean.data.add_(nat.grad.rename(*nat.names).align_as(mean), alpha=lr)
            new_nats = self.mean2nat(*means)

            for old_nat, new_nat in zip(self.named_nats, new_nats):
                old_nat.data.copy_(new_nat.align_as(old_nat))

class TiltedNormal(Tilted, NormalMixin):
    pass
class TiltedMvNormal(Tilted, MvNormalMixin):
    pass
class TiltedBernoulli(Tilted, BernoulliMixin):
    pass
class TiltedPoisson(Tilted, PoissonMixin):
    pass
class TiltedExponential(Tilted, ExponentialMixin):
    pass
class TiltedDirichlet(Tilted, DirichletMixin):
    pass
class TiltedBeta(Tilted, BetaMixin):
    pass
class TiltedGamma(Tilted, GammaMixin):
    pass
