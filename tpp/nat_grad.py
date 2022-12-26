import torch as t
import torch.nn as nn
from .dist import *
from .utils import *
from .qmodule import QModule

def identity(x):
    return x

def tuple_assert_allclose(xs, ys):
    for (x, y) in zip(xs, ys):
        assert t.allclose(x, y, atol=1E-5)

def grad_digamma(x):
    return t.special.polygamma(1, x)

def inverse_digamma(y):
    """
    Solves y = digamma(x)
    or computes x = digamma^{-1}(y)
    Appendix C in https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
    Works very well assuming the x's you start with are all positive
    """
    x_init_for_big_y = y.exp()+0.5
    x_init_for_small_y = -t.reciprocal(y-t.digamma(t.ones(())))
    x = t.where(y>-2.22, x_init_for_big_y, x_init_for_small_y)
    for _ in range(6):
        x = x - (t.digamma(x) - y)/grad_digamma(x)
    return x

class NG(QModule):
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
    def __init__(self, platesizes=None, sample_shape=(), init_conv=None):
        super().__init__()
        if init_conv is None:
            init_conv = self.default_init_conv
        init_means = self.conv2mean(*init_conv)

        if platesizes is None:
            platesizes = {}
        shape = [*platesizes.values(), *sample_shape]
        names = [*platesizes.keys(), *(len(sample_shape) * [None])]

        self.meannames = tuple(f'mean_{i}' for i in range(len(self.sufficient_stats)))
        for (meanname, init_mean) in zip(self.meannames, init_means):
            self.register_buffer(meanname, t.full(shape, init_mean).rename(*names))

        self.natnames = tuple(f'nat_{i}' for i in range(len(self.sufficient_stats)))
        for natname in self.natnames:
            self.register_parameter(natname, nn.Parameter(t.zeros(shape).rename(*names)))
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

    def forward(self):
        return self.dist(*self.nat2conv(*self.dim_nats))

    def update(self, lr):
        with t.no_grad():
            for (mean, nat) in zip(self.named_means, self.named_nats):
                mean.data.add_(nat.grad.rename(*nat.names).align_as(mean), alpha=lr)
        self.reset_nats()

    def reset_nats(self):
        with t.no_grad():
            new_nats = self.mean2nat(*self.named_means)
            for old_nat, new_nat in zip(self.named_nats, new_nats):
                old_nat.data.copy_(new_nat.align_as(old_nat))
                old_nat.grad=None

    @classmethod
    def mean2nat(cls, *mean):
        return cls.conv2nat(*cls.mean2conv(*mean))
    @classmethod
    def nat2mean(cls, *nat):
        return cls.conv2mean(*cls.nat2conv(*nat))
    @classmethod
    def test(self, N):
        conv = self.test_conv(N)
        mean = self.conv2mean(*conv)
        nat  = self.conv2nat(*conv)

        tuple_assert_allclose(conv, self.mean2conv(*mean))
        tuple_assert_allclose(conv, self.nat2conv(*nat))

        tuple_assert_allclose(nat,  self.mean2nat(*mean))
        tuple_assert_allclose(mean, self.nat2mean(*nat))

class NGNormal(NG):
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
    def conv2nat(loc, scale):
        prec = 1/scale**2
        mu_prec = loc * prec
        return mu_prec, -0.5*prec
    @staticmethod
    def nat2conv(mu_prec, minus_half_prec):
        prec = -2*minus_half_prec
        mu = mu_prec / prec
        scale = prec.rsqrt()
        return mu, scale
    @staticmethod
    def test_conv(N):
        return (t.randn(N), t.randn(N).exp())


@staticmethod
def identity_conv(*args):
    return args
class NG_NEF(NG):
    """
    For Natural Exponential Families, see:
    https://en.wikipedia.org/wiki/Natural_exponential_family
    """
    sufficient_stats = (identity,)
    conv2mean = identity_conv
    mean2conv = identity_conv
class NGBernoulli(NG_NEF):
    dist = staticmethod(Bernoulli)
    @staticmethod
    def test_conv(N):
        return (t.rand(N),)
class NGPoisson(NG_NEF):
    dist = staticmethod(Poisson)
    @staticmethod
    def test_conv(N):
        return (t.randn(N).exp(),)

class NGExponential(NG):
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

class NGDirichlet(NG):
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

class NGGamma(NG):
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
