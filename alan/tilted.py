import torch as t
import torch.nn as nn
from .dist import *
from .utils import *
from .alan_module import AlanModule
from .exp_fam_mixin import *
from . import model

class Tilted(model.Model):
    """
    Based on computing:
    Delta m = E_P[f(x)] - E_Q[f(x)]
    where P is the true posterior.
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

        self.Japprox_names = tuple(f'Japprox_{i}' for i in range(len(self.sufficient_stats)))
        self.Jpost_names = tuple(f'Jpost_{i}' for i in range(len(self.sufficient_stats)))
        for Jname in [*self.Japprox_names, *self.Jpost_names]:
            self.register_parameter(Jname, nn.Parameter(t.zeros(shape).rename(*names)))

        self.platenames = tuple(platesizes.keys())

    @property
    def dim_nats(self):
        return [getattr(self, natname) for natname in self.natnames]
    @property
    def named_nats(self):
        return [self.get_named_tensor(natname) for natname in self.natnames]

    @property
    def dim_Japproxs(self):
        return [getattr(self, Japprox_name) for Japprox_name in self.Japprox_names]
    @property
    def named_Japproxs(self):
        return [self.get_named_tensor(Japprox_name) for Japprox_name in self.Japprox_names]

    @property
    def dim_Jposts(self):
        return [getattr(self, Jpost_name) for Jpost_name in self.Jpost_names]
    @property
    def named_Jposts(self):
        return [self.get_named_tensor(Jpost_name) for Jpost_name in self.Jpost_names]

    def tilted(self, *args, **kwargs):
        with t.no_grad():
            prior = self.dist(*args, **kwargs)
            prior_convs = self.canonical_conv(**prior.dim_args)
            prior_nats = self.conv2nat(**prior_convs)

            Q_nats = tuple(prior+dn for (prior, dn) in zip(prior_nats, self.dim_nats))
            Q_means = self.nat2mean(*Q_nats)

        def inner(sample):
            #Check the dimensions of sample are as expected.
            if len(sample.dims) != len(self.platenames) + 1:
                raise Exception(f"Unexpected sample dimensions.  We expected {self.platenames}, with an extra K-dimension.  We got {sample.dims}.  If the K-dimension is missing, you may have set multi_sample=False, which is not compatible with ML2 proposals/approximate posteriors")
            J_posts  = sum(sum_non_dim(J*f(sample)) for (J, f)         in zip(self.dim_Jposts,   self.sufficient_stats))
            J_approx = sum(sum_non_dim(J*Q_mean)    for (J, Q_mean)    in zip(self.dim_Japproxs, Q_means))
            return (J_posts + J_approx)

        return self.dist(**self.nat2conv(*Q_nats), extra_log_factor=inner)

    def P(self, tr, *args, **kwargs):
        tr(self.dist(*args, **kwargs))
    def Q(self, tr, *args, **kwargs):
        tr(self.tilted(*args, **kwargs))

    def _update(self, lr):
        with t.no_grad():
            #post_ms   = tuple(J.grad for J in self.named_Jposts)
            #approx_ms = tuple(J.grad for J in self.named_Japproxs)
            post_ms   = tuple(self.get_named_grad(name) for name in self.Jpost_names)
            approx_ms = tuple(self.get_named_grad(name) for name in self.Japprox_names)
            new_ms    = tuple(lr*post_m + (1-lr)*approx_m for (post_m, approx_m) in zip(post_ms, approx_ms))
            new_ns    = self.mean2nat(*new_ms)
            old_ns    = self.mean2nat(*approx_ms)
            dns       = tuple(new_n - old_n for (new_n, old_n) in zip(new_ns, old_ns))

            for (n, dn) in zip(self.named_nats, dns):
                n.data.add_(dn)

    def local_parameters(self):
        return []
        
        

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
