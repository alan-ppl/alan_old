import torch as t
import torch.nn as nn
from .exp_fam_conversions import conv_dict
from .dist import *
from .utils import *
from .qmodule import QModule

def identity(x):
    return x

class ML(QModule):
    def __init__(self, platesizes=None, sample_shape=(), init_conv=None):
        super().__init__()
        if init_conv is None:
            init_conv = self.default_init_conv
        init_means = self.conv2mean(*init_conv)

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
            raise Exception("One of the Js is non-zero, presumably this is because ...")

    def forward(self):
        return self.dist(*self.mean2conv(*self.dim_means), extra_log_factor=self.extra_log_factor)

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

class MLNormal(ML):
    dist = staticmethod(Normal)
    sufficient_stats = (identity, t.square)
    default_init_conv = (0., 1.)
    @staticmethod
    def conv2mean(loc, scale):
        Ex  = loc
        Ex2 = loc**2 + scale**2
        return Ex, Ex2
    @staticmethod
    def mean2conv(Ex, Ex2):
        loc   = Ex 
        scale = (Ex2 - loc**2).sqrt()
        return loc, scale
    @staticmethod
    def test_conv(N):
        return (t.randn(N), t.randn(N).exp())


@staticmethod
def identity_conv(*args):
    return args
class ML_NEF(ML):
    """
    For Natural Exponential Families, see:
    https://en.wikipedia.org/wiki/Natural_exponential_family
    """
    sufficient_stats = (identity,)
    conv2mean = identity_conv
    mean2conv = identity_conv
class BernoulliConversions(ML_NEF):
    dist = staticmethod(Bernoulli)
    def test_conv(self, N):
        return (t.rand(N),)
class PoissonConversions(ML_NEF):
    dist = staticmethod(Poisson)
    def test_conv(self, N):
        return (t.randn(N).exp(),)

class MLExponential(ML):
    dist = staticmethod(Exponential)
    sufficient_stats = (identity,)
    @staticmethod
    def conv2mean(mean):
        return (t.reciprocal(mean),)
    @staticmethod
    def mean2conv(mean):
        return (t.reciprocal(mean),)
    @staticmethod
    def test_conv(N):
        return (t.randn(N).exp(),)

class MLDirichlet(ML):
    dist = staticmethod(Dirichlet)
    sufficient_stats = (t.log,)

    @staticmethod
    def conv2mean(alpha):
        return (t.digamma(alpha) - t.digamma(alpha.sum(-1, keepdim=True)),)

    @staticmethod
    def mean2conv(logp):
        """
        Methods from https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
        """
        alpha = t.ones_like(logp)
        #Initialize with fixed point iterations from Eq. 9 that are slow, but guaranteed to converge
        for _ in range(5):
            alpha = inverse_digamma(t.digamma(alpha.sum(-1, keepdim=True)) + logp)

        #Clean up with a few fast but unstable Newton's steps (Eq. 15-18)
        for _ in range(6):
            sum_alpha = alpha.sum(-1, keepdim=True)
            g = (t.digamma(sum_alpha) - t.digamma(alpha) + logp) #Eq. 6
            z = grad_digamma(sum_alpha)
            q = - grad_digamma(alpha)
            b = (g/q).sum(-1, keepdim=True) / (1/z + (1/q).sum(-1, keepdim=True))
            alpha = alpha - (g - b)/q
        return (alpha,)

    @staticmethod
    def test_conv(N):
        return (t.randn(N, 4).exp(),)

class MLGamma(ML):
    dist = staticmethod(Gamma)
    sufficient_stats = (t.log, identity)
    @staticmethod
    def conv2mean(alpha, beta):
        #Tested by sampling
        return (-t.log(beta) + t.digamma(alpha), alpha/beta)
    @staticmethod
    def mean2conv(Elogx, Ex):
        """
        Generalised Newton's method from Eq. 10 in https://tminka.github.io/papers/minka-gamma.pdf
        Rewrite as:
        1/a^new = 1/a (1 + num / a (1/a + grad_digamma(a)))
        1/a^new = 1/a (1 + num / (1 + a grad_digamma(a)))
        a^new   = a / (1 + num / (1 + a grad_digamma(a)))
        """
        logEx = Ex.log()
        diff = (Elogx - logEx)
        alpha = - 0.5 / diff
        for _ in range(6):
            num = diff + alpha.log() - t.digamma(alpha)
            denom = 1 - alpha * grad_digamma(alpha)
            alpha = alpha * t.reciprocal(1 + num/denom)
        beta = alpha / Ex 
        return (alpha, beta)
    @staticmethod
    def test_conv(N):
        return (t.randn(N).exp(),t.randn(N).exp())
