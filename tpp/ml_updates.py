import torch as t
import torch.nn as nn
from .dist import *
from .utils import *
from .qmodule import QModule
from .exp_fam_mixin import *

class ML(QModule):
    def __init__(self, platesizes=None, sample_shape=(), init_conv=None):
        super().__init__()
        if init_conv is None:
            init_conv = self.default_init_conv
        init_means = self.conv2mean(**init_conv)

        if platesizes is None:
            platesizes = {}
        shape = [*platesizes.values(), *sample_shape]
        names = [*platesizes.keys(), *(len(sample_shape) * [None])]

        self.meannames = tuple(f'm{i}' for i in range(len(self.sufficient_stats)))
        for (meanname, init_mean) in zip(self.meannames, init_means):
            self.register_buffer(meanname, t.full(shape, init_mean).rename(*names))

        self.Jnames = tuple(f'J{i}' for i in range(len(self.sufficient_stats)))
        for Jname in self.Jnames:
            self.register_parameter(Jname, nn.Parameter(t.zeros(shape).rename(*names)))

        self.platenames = tuple(platesizes.keys())

    @property
    def dim_means(self):
        return [getattr(self, meanname) for meanname in self.meannames]
    @property
    def named_means(self):
        return [self.get_named_tensor(meanname) for meanname in self.meannames]

    @property
    def dim_Js(self):
        return [getattr(self, Jname) for Jname in self.Jnames]
    @property
    def named_Js(self):
        return [self.get_named_tensor(Jname) for Jname in self.Jnames]

    @property
    def named_grads(self):
        return [J.grad.rename(*J.names) for J in self.named_Js]

    def check_J_zeros(self):
        if not all((J==0).all() for J in self.named_Js):
            raise Exception("One of the Js is non-zero. Presumably this is because one of the Js has been given to an optimizer as a parameter.  The solution is to use model.parameters() to extract parameters, as this avoids Js, rather than e.g. Q.parameters()")

    def forward(self):
        return self.dist(**self.mean2conv(*self.dim_means), extra_log_factor=self.extra_log_factor)

    def extra_log_factor(self, sample):
        #Check the dimensions of sample are as expected.
        if len(sample.dims) != len(self.platenames) + 1:
            raise Exception(f"Unexpected sample dimensions.  We expected {self.platenames}, with an extra K-dimension.  We got {sample.dims}.  If the K-dimension is missing, you may have set multi_sample=False, which is not compatible with ML proposals/approximate posteriors")
        #The factor comes in through log Q, so must be negated!
        return -sum(sum_non_dim(J*f(sample)) for (J, f) in zip(self.dim_Js, self.sufficient_stats))

    def init(self):
        with t.no_grad():
            for (m, g) in zip(self.named_means, self.named_grads):
                m.data.copy_(g.align_as(m))

    def update(self, lr):
        self.check_J_zeros()
        with t.no_grad():
            for (m, g) in zip(self.named_means, self.named_grads):
                m.data.mul_(1-lr).add_(g.align_as(m), alpha=lr)

class MLNormal(ML, NormalMixin):
    pass
class MLMvNormal(ML, MvNormalMixin):
    pass
class MLBernoulli(ML, BernoulliMixin):
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
