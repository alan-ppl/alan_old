import torch as t
import torch.nn as nn
from .dist import *
from .utils import *
from .qmodule import QModule
from .exp_fam_mixin import *

class Tilted(QModule):
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

    def __init__(self, platesizes=None, sample_shape=()):
        super().__init__()

        if platesizes is None:
            platesizes = {}
        shape = [*platesizes.values(), *sample_shape]
        names = [*platesizes.keys(), *(len(sample_shape) * [None])]

        self.natnames = tuple(f'nat_{i}' for i in range(len(self.sufficient_stats)))
        for natname in self.natnames:
            self.register_buffer(natname, t.zeros(shape).rename(*names))

        self.meannames = tuple(f'mean_{i}' for i in range(len(self.sufficient_stats)))
        for meanname in self.meannames:
            self.register_parameter(meanname, nn.Parameter(t.zeros(shape).rename(*names)))

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

    def forward(self, prior):
        with t.no_grad():
            if not isinstance(prior, self.dist):
                raise(f"{type(self)} can only be combined with {type(self.dist)} distributions")
            prior_convs = self.canonical_conv(**prior.dim_args)
            prior_nats = self.conv2nat(**prior_convs)
            post_nats = tuple(prior+dn for (prior, dn) in zip(prior_nats, self.dim_nats))
            post_means = self.nat2mean(*post_nats)
            
        for m in self.named_means:
            assert (m==0).all()
            assert (m.grad is None) or (m.grad==0).all()
        _post_means = tuple(pm.detach() + m for (pm, m) in zip(post_means, self.dim_means))
        return self.dist(**self.mean2conv(*_post_means))

    def update(self, lr):
        with t.no_grad():
            for (mean, nat) in zip(self.named_means, self.named_nats):
                nat.add_(mean.grad.rename(*mean.names).align_as(nat), alpha=lr)


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
