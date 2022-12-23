import torch as t
import torch.autograd.forward_ad as fwAD
from .dist import *

Ex     = lambda x: x
Ex2    = lambda x: x**2
Elogx  = lambda x: x.log()

def identity(self, *args):
    return args

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

class AbstractConversions():
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
    def mean2nat(self, *mean):
        return self.conv2nat(*self.mean2conv(*mean))
    def nat2mean(self, *nat):
        return self.conv2mean(*self.nat2conv(*nat))

    def test(self, N):
        conv = self.test_conv(N)
        mean = self.conv2mean(*conv)
        nat  = self.conv2nat(*conv)

        tuple_assert_allclose(conv, self.mean2conv(*mean))
        tuple_assert_allclose(conv, self.nat2conv(*nat))

        tuple_assert_allclose(nat,  self.mean2nat(*mean))
        tuple_assert_allclose(mean, self.nat2mean(*nat))

class AbstractNEFConversions(AbstractConversions):
    """
    For Natural Exponential Families, see:
    https://en.wikipedia.org/wiki/Natural_exponential_family
    """
    sufficient_stats = (Ex,)
    conv2nat  = identity
    nat2conv  = identity
    conv2mean = identity
    mean2conv = identity
class BernoulliConversions(AbstractNEFConversions):
    dist = Bernoulli
    def test_conv(self, N):
        return (t.rand(N),)

class PoissonConversions(AbstractNEFConversions):
    dist = Poisson
    def test_conv(self, N):
        return (t.randn(N).exp(),)

class NormalConversions(AbstractConversions):
    dist = Normal
    sufficient_stats = (Ex, Ex2)
    def conv2mean(self, loc, scale):
        Ex  = loc
        Ex2 = loc**2 + scale**2
        return Ex, Ex2
    def mean2conv(self, Ex, Ex2):
        loc   = Ex 
        scale = (Ex2 - loc**2).sqrt()
        return loc, scale

    def conv2nat(self, loc, scale):
        prec = 1/scale**2
        mu_prec = loc * prec
        return prec, mu_prec
    def nat2conv(self, prec, mu_prec):
        mu = mu_prec / prec
        scale = prec.rsqrt()
        return mu, scale

    def test_conv(self, N):
        return (t.randn(N), t.randn(N).exp())

class ExponentialConversions(AbstractConversions):
    dist = Exponential
    sufficient_stats = (Ex, Ex2)
    def conv2mean(self, mean):
        return (t.reciprocal(mean),)
    def mean2conv(self, mean):
        return (t.reciprocal(mean),)
    
    nat2conv = identity
    conv2nat = identity

    def test_conv(self, N):
        return (t.randn(N).exp(),)

class DirichletConversions(AbstractConversions):
    dist = Dirichlet
    sufficient_stats = (Elogx,)

    def nat2conv(self, nat):
        return ((nat+1),)
    def conv2nat(self, alpha):
        return ((alpha-1),)

    def conv2mean(self, alpha):
        return (t.digamma(alpha) - t.digamma(alpha.sum(-1, keepdim=True)),)
    def mean2conv(self, logp):
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

    def test_conv(self, N):
        return (t.randn(N, 4).exp(),)

class GammaConversions(AbstractConversions):
    dist = Gamma
    sufficient_stats = (Elogx, Ex)

    def conv2nat(self, alpha, beta):
        return (alpha-1, -beta)
    def nat2conv(self, n1, n2):
        return (n1+1, -n2)

    def conv2mean(self, alpha, beta):
        #Tested by sampling
        return (-t.log(beta) + t.digamma(alpha), alpha/beta)
    def mean2conv(self, Elogx, Ex):
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

    def test_conv(self, N):
        return (t.randn(N).exp(),t.randn(N).exp())

conv_dict = {
    Bernoulli: BernoulliConversions(),
    Poisson: PoissonConversions(),
    Normal: NormalConversions(),
    Exponential: ExponentialConversions(),
    Dirichlet: DirichletConversions(),
    Gamma: GammaConversions(),
}
    

if __name__ == "__main__":
    N = 10
    BernoulliConversions().test(N)
    PoissonConversions().test(N)
    NormalConversions().test(N)
    ExponentialConversions().test(N)
    DirichletConversions().test(N)
    GammaConversions().test(N)
