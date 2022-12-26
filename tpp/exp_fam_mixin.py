import torch as t
import torch.autograd.forward_ad as fwAD
from .dist import *

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

def tuple_assert_allclose(xs, ys):
    for (x, y) in zip(xs, ys):
        assert t.allclose(x, y, atol=1E-5)

class AbstractMixin():
    """
    Must provide methods interconvert natural <-> conventional <-> mean parameters, i.e.
    self.conv2nat(self, *conv)
    self.nat2conv(self, *nat)

    self.conv2mean(self, *conv)
    self.mean2conv(self, *mean)

    We provide a default for natural <-> mean parameters by going through the conventional
    parameters
    """
    #Interchange mean and natural parameters by going through conventional parameters
    @classmethod
    def mean2nat(cls, *mean):
        return cls.conv2nat(*cls.mean2conv(*mean))
    @classmethod
    def nat2mean(cls, *nat):
        return cls.conv2mean(*cls.nat2conv(*nat))

    @classmethod
    def test(cls, N):
        conv = cls.test_conv(N)
        mean = cls.conv2mean(*conv)
        nat  = cls.conv2nat(*conv)

        tuple_assert_allclose(conv, cls.mean2conv(*mean))
        tuple_assert_allclose(conv, cls.nat2conv(*nat))

        tuple_assert_allclose(nat,  cls.mean2nat(*mean))
        tuple_assert_allclose(mean, cls.nat2mean(*nat))

@staticmethod
def identity_conv(*args):
    return args
def identity(x):
    return x
class AbstractNEFMixin(AbstractMixin):
    """
    For Natural Exponential Families, see:
    https://en.wikipedia.org/wiki/Natural_exponential_family
    """
    sufficient_stats = (identity,)
    conv2nat  = identity_conv
    nat2conv  = identity_conv
    conv2mean = identity_conv
    mean2conv = identity_conv
class BernoulliMixin(AbstractNEFMixin):
    dist = staticmethod(Bernoulli)
    @staticmethod
    def test_conv(N):
        return (t.rand(N),)

class PoissonMixin(AbstractNEFMixin):
    dist = staticmethod(Poisson)
    @staticmethod
    def test_conv(N):
        return (t.randn(N).exp(),)

class NormalMixin(AbstractMixin):
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

class ExponentialMixin(AbstractMixin):
    dist = staticmethod(Exponential)
    sufficient_stats = (identity,)
    @staticmethod
    def conv2mean(mean):
        return (t.reciprocal(mean),)
    @staticmethod
    def mean2conv(mean):
        return (t.reciprocal(mean),)
    
    nat2conv = identity_conv
    conv2nat = identity_conv

    @staticmethod
    def test_conv(N):
        return (t.randn(N).exp(),)

class DirichletMixin(AbstractMixin):
    dist = staticmethod(Dirichlet)
    sufficient_stats = (t.log,)

    @staticmethod
    def nat2conv(nat):
        return ((nat+1),)
    @staticmethod
    def conv2nat(alpha):
        return ((alpha-1),)

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

class BetaMixin(AbstractMixin):
    dist = staticmethod(Beta)
    sufficient_stats = (t.log, lambda x: t.log(1-x))
    
    @staticmethod
    def conv2nat(alpha, beta):
        return (alpha-1, beta-1)
    @staticmethod
    def nat2conv(nat_0, nat_1):
        return (nat_0+1, nat_1+1)

    @staticmethod
    def conv2mean(alpha, beta):
        norm = t.digamma(alpha + beta)
        return (t.digamma(alpha) - norm, t.digamma(beta) - norm)
    @staticmethod
    def mean2conv(Elogx, Elog1mx):
        logp = t.stack([Elogx, Elog1mx], -1)
        alpha = DirichletMixin.mean2conv(logp)[0]
        return alpha[..., 0], alpha[..., 1]
    @staticmethod
    def test_conv(N):
        return (t.randn(N).exp(), t.randn(N).exp())

    

class GammaMixin(AbstractMixin):
    dist = staticmethod(Gamma)
    sufficient_stats = (t.log, identity)

    @staticmethod
    def conv2nat(alpha, beta):
        return (alpha-1, -beta)
    @staticmethod
    def nat2conv(n1, n2):
        return (n1+1, -n2)

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

class InverseGammaMixin(AbstractMixin):
    dist = staticmethod(Gamma)
    sufficient_stats = (t.log, t.reciprocal)

    @staticmethod
    def conv2nat(alpha, beta):
        return (-alpha-1, -beta)
    @staticmethod
    def nat2conv(nat0, nat1):
        return (-nat0-1, -nat1)

    @staticmethod
    def conv2mean(alpha, beta):
        #From Wikipedia (Inverse Gamma: Properties)
        return (t.log(beta) - t.digamma(alpha), alpha/beta)
    @staticmethod
    def mean2conv(mean_0, mean_1):
        return GammaMixin.mean2conv(-mean_0, mean_1)

    @staticmethod
    def test_conv(N):
        return (t.randn(N).exp(),t.randn(N).exp())

