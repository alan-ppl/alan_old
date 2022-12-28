import torch as t
import torch.nn as nn
from .dist import *
from .utils import *
from .qmodule import QModule
from .exp_fam_mixin import *

class NG(QModule):
    """
    This really is NG, though is slightly different from the usual NG-VI setting.
    In particular, in essence, we compute E_P[grad log Q], where P is our reweighted
    posterior approximation.

    In this case, we take the gradient wrt m, and for ease of understanding, we take
    P(z) = g(z) exp(n_0 T(z) - A(n_0))
    Q(z) = g(z) exp(n   T(z) - A(n))

    grad_m E_P[log Q] = grad_m E_P[n T(z) - A(n)]
                      = grad_m [n m_0 - A(n)]
                      = grad_m [n] m_0 - grad_m[A(n)]
                      = F^-1 (m_0 - m)
    We apply this update to natural params, nn.
    Only works when we can differentiate through e.g. mean2conv.
    """

    def __init__(self, platesizes=None, sample_shape=(), init_conv=None):
        super().__init__()
        if init_conv is None:
            init_conv = self.default_init_conv
        init_nats = self.conv2nat(**init_conv)

        if platesizes is None:
            platesizes = {}
        shape = [*platesizes.values(), *sample_shape]
        names = [*platesizes.keys(), *(len(sample_shape) * [None])]

        self.natnames = tuple(f'nat_{i}' for i in range(len(self.sufficient_stats)))
        for (natname, init_nat) in zip(self.natnames, init_nats):
            self.register_buffer(natname, t.full(shape, init_nat).rename(*names))

        self.meannames = tuple(f'mean_{i}' for i in range(len(self.sufficient_stats)))
        for meanname in self.meannames:
            self.register_parameter(meanname, nn.Parameter(t.zeros(shape).rename(*names)))
        self.reset_means()

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

    def forward(self):
        return self.dist(**self.mean2conv(*self.dim_means))

    def update(self, lr):
        with t.no_grad():
            for (mean, nat) in zip(self.named_means, self.named_nats):
                nat.data.add_(mean.grad.rename(*mean.names).align_as(nat), alpha=lr)
        self.reset_means()

    def reset_means(self):
        with t.no_grad():
            new_means = self.nat2mean(*self.named_nats)
            for old_mean, new_mean in zip(self.named_means, new_means):
                old_mean.data.copy_(new_mean.align_as(old_mean))

class NGNormal(NG, NormalMixin):
    pass
class NGMvNormal(NG, MvNormalMixin):
    pass
class NGBernoulli(NG, BernoulliMixin):
    pass
class NGPoisson(NG, PoissonMixin):
    pass
class NGExponential(NG, ExponentialMixin):
    pass
