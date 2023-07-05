import torch as t
import torch.autograd.forward_ad as fwAD
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from torch.nn.functional import threshold
from .dist import *

import alan.postproc as pp

Tensor = (functorch.dim.Tensor, t.Tensor)

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

def dict_assert_allclose(xs, ys):
    assert set(xs.keys()) == set(ys.keys())
    for key in xs:
        assert t.allclose(xs[key], ys[key], atol=1E-5)

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
        return cls.conv2nat(**cls.mean2conv(*mean))
    @classmethod
    def nat2mean(cls, *nat):
        return cls.conv2mean(**cls.nat2conv(*nat))

    @classmethod
    def test(cls, N):
        conv = cls.test_conv(N)
        mean = cls.conv2mean(**conv)
        nat  = cls.conv2nat(**conv)

        dict_assert_allclose(conv, cls.mean2conv(*mean))
        dict_assert_allclose(conv, cls.nat2conv(*nat))

        tuple_assert_allclose(nat,  cls.mean2nat(*mean))
        tuple_assert_allclose(mean, cls.nat2mean(*nat))

@staticmethod
def identity_conv(*args):
    return args
def identity(x):
    return x

def inverse_sigmoid(y):
    """
    y = 1/(1+e^(-x))
    1+e^(-x) = 1/y
    e^(-x) = 1/y - 1
    -x = log(1/y - 1)
    x = -log(1/y - 1)
    """
    return -t.log(t.abs(1/y - 1) + 1e-50)

class BernoulliMixin(AbstractMixin):
    """
    P(x) = p^x (1-p)^(1-x)
    log P(x) = x log p + (1-x) log (1-p)
    log P(x) = log (1-p) + x log (p/(1-p))
    log P(x) = log (1-p) + x logits

    Thus, logits is the natural parameter.
    However, we could choose to use logits or probs as the conventional parameter
    """
    dist = staticmethod(Bernoulli)
    sufficient_stats = (identity,)

class BernoulliLogitsMixin(BernoulliMixin):
    #Conventional and natural parameters are both logits
    default_init_conv = {'logits':t.zeros(())}

    @staticmethod
    def conv2nat(logits):
        return (logits,)
    @staticmethod
    def nat2conv(logits):
        return {'logits': logits}

    @staticmethod
    def conv2mean(logits):
        return (t.sigmoid(logits),)
    @staticmethod
    def mean2conv(probs):
        return {'logits': inverse_sigmoid(probs)}

    @staticmethod
    def canonical_conv(logits=None, probs=None):
        assert (probs is None) != (logits is None)
        return {'logits': inverse_sigmoid(probs) if probs is not None else logits}

    @staticmethod
    def test_conv(N):
        return {'logits': t.randn(N)}

class BernoulliProbsMixin(BernoulliMixin):
    #Conv is prob, nat is logits
    @staticmethod
    def conv2nat(probs):
        return (inverse_sigmoid(probs),)
    @staticmethod
    def nat2conv(logits):
        return {'probs': t.sigmoid(logits)}

    @staticmethod
    def conv2mean(probs):
        return (probs,)
    @staticmethod
    def mean2conv(probs):
        return {'probs': probs}

    @staticmethod
    def canonical_conv(logits=None, probs=None):
        assert (probs is None) != (logits is None)
        return {'probs': t.sigmoid(logits) if (logits is not None) else logits}

    @staticmethod
    def test_conv(N):
        return {'probs': t.rand(N)}



class PoissonMixin(AbstractMixin):
    dist = staticmethod(Poisson)
    sufficient_stats = (identity,)
    @staticmethod
    def conv2mean(rate):
        return (rate,)
    @staticmethod
    def mean2conv(rate):
        return {'rate': rate}

    @staticmethod
    def conv2nat(rate):
        return (t.log(rate),)
    @staticmethod
    def nat2conv(log_rate):
        return {'rate': t.exp(log_rate)}

    @staticmethod
    def test_conv(N):
        return {'rate': t.randn(N).exp()}

    @staticmethod
    def canonical_conv(rate):
        return {'rate': rate.exp()}

# class NormalMixin(AbstractMixin):
#     dist = staticmethod(Normal)
#     sufficient_stats = (identity, t.square)
#     default_init_conv = {'loc': 0., 'scale': 1.}
#     @staticmethod
#     def conv2mean(loc, scale):
#         Ex  = loc
#         Ex2 = loc**2 + scale**2
#         return Ex, Ex2
#     @staticmethod
#     def mean2conv(Ex, Ex2):
#         loc   = Ex
#         # print(Ex2)
#         # print(loc)
#         # print(Ex2 - loc**2)
#         scale = threshold(Ex2 - loc**2, 0 ,1e-25).sqrt()
#         return {'loc': loc, 'scale': scale}

#     @staticmethod
#     def conv2nat(loc, scale):


#         prec = 1/threshold(scale, 0 ,1e-25)
#         mu_prec = loc * prec
#         return mu_prec, -0.5*prec

#     @staticmethod
#     def nat2conv(mu_prec, minus_half_prec):
#         prec = -2*minus_half_prec
#         loc = mu_prec / threshold(prec, 0 ,1e-25)
#         scale = threshold(prec, 0 ,1e-25).rsqrt()
#         return {'loc': loc, 'scale': scale}

#     @staticmethod
#     def canonical_conv(loc, scale):
#         ## exp so that its positive, is this the right place to do this?
#         return {'loc': loc, 'scale': scale}

#     @staticmethod
#     def test_conv(N):
#         return {'loc': t.randn(N), 'scale': t.randn(N).exp()}

class NormalMixin(AbstractMixin):
    dist = staticmethod(Normal)
    sufficient_stats = (identity, t.square)
    default_init_conv = {'loc': 0., 'scale': 1.}
    @staticmethod
    def conv2mean(loc, scale):
        Ex  = loc
        Ex2 = loc**2 + scale**2

        return Ex, Ex2
    
    @staticmethod
    def sample2mean(sample,index):
        name = list(sample.keys())[index]
        sample = {name:sample[name]}
        Ex  = pp.mean(sample)[name]
        Ex2 = pp.mean(pp.square(sample))[name]

        return [Ex, Ex2]
    
    @staticmethod
    def mean2conv(Ex, Ex2):
        loc   = Ex
        # print(Ex2 - loc**2)
        scale = (threshold(Ex2 - loc**2,1,1)).sqrt() 
        # scale = scale + (1e-20)*(scale==0)
        # scale = (Ex2 - loc**2).sqrt() 
        # Try this:
        # a = Ex2 - loc**2
        # A = a + (-a + 1e-20)*(a<0)
        # scale = A.sqrt()
        # print(Ex2 - loc**2 <= 0)
        # print(scale)
        return {'loc': loc, 'scale': scale}

    @staticmethod
    def conv2nat(loc, scale):
        prec = 1/scale
        mu_prec = loc * prec
        return mu_prec, -0.5*prec

    @staticmethod
    def nat2conv(mu_prec, minus_half_prec):
        prec = -2*minus_half_prec
        loc = mu_prec / prec
        scale = prec.rsqrt()
        return {'loc': loc, 'scale': scale}

    @staticmethod
    def canonical_conv(loc, scale):
        return {'loc': loc, 'scale': scale}

    @staticmethod
    def test_conv(N):
        return {'loc': t.randn(N), 'scale': t.randn(N).exp()}
    
class ExponentialMixin(AbstractMixin):
    dist = staticmethod(Exponential)
    sufficient_stats = (identity,)
    @staticmethod
    def conv2mean(rate):
        return (t.reciprocal(rate),)
    @staticmethod
    def mean2conv(mean):
        return {'rate': t.reciprocal(mean)}

    @staticmethod
    def nat2conv(nat):
        return {'rate': -nat}
    @staticmethod
    def conv2nat(rate):
        return (-rate,)

    @staticmethod
    def test_conv(N):
        return {'rate': t.randn(N).exp()}

    @staticmethod
    def canonical_conv(rate):
        return {'rate': rate}


class DirichletMixin(AbstractMixin):
    dist = staticmethod(Dirichlet)
    sufficient_stats = (t.log,)

    @staticmethod
    def nat2conv(nat):
        return {'concentration': (nat+1)}
    @staticmethod
    def conv2nat(concentration):
        return ((concentration-1),)

    @staticmethod
    def conv2mean(concentration):
        return (t.digamma(concentration) - t.digamma(concentration.sum(-1, keepdim=True)),)
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
        return {'concentration': alpha}

    @staticmethod
    def test_conv(N):
        return {'concentration': t.randn(N, 4).exp()}

    @staticmethod
    def canonical_conv(concentration):
        return {'concentration': concentration}

class BetaMixin(AbstractMixin):
    dist = staticmethod(Beta)
    sufficient_stats = (t.log, lambda x: t.log(1-x))

    @staticmethod
    def conv2nat(concentration1, concentration0):
        return (concentration1-1, concentration0-1)
    @staticmethod
    def nat2conv(nat_0, nat_1):
        return {'concentration1':nat_0+1, 'concentration0': nat_1+1}

    @staticmethod
    def conv2mean(concentration1, concentration0):
        norm = t.digamma(concentration1 + concentration0)
        return (t.digamma(concentration1) - norm, t.digamma(concentration0) - norm)
    @staticmethod
    def mean2conv(Elogx, Elog1mx):
        logp = t.stack([Elogx, Elog1mx], -1)
        c = DirichletMixin.mean2conv(logp)['concentration']
        return {'concentration1': c[..., 0], 'concentration0': c[..., 1]}
    @staticmethod
    def test_conv(N):
        return {'concentration1': t.randn(N).exp(),'concentration0': t.randn(N).exp()}

    @staticmethod
    def canonical_conv(concentration1, concentration0):
        return {'concentration1': concentration1, 'concentration0': concentration0}



class GammaMixin(AbstractMixin):
    """
    concentration == alpha
    rate == beta
    """
    dist = staticmethod(Gamma)
    sufficient_stats = (t.log, identity)
    default_init_conv = {'concentration':t.tensor(2, dtype=t.float64), 'rate':t.tensor(1, dtype=t.float64)}

    @staticmethod
    def conv2nat(concentration, rate):
        alpha = concentration
        beta = rate
        return (alpha-1, -beta)
    @staticmethod
    def nat2conv(n1, n2):
        return {'concentration': n1+1, 'rate': -n2}

    @staticmethod
    def conv2mean(concentration, rate):
        #Tested by sampling
        alpha = concentration
        beta = rate
        return (-t.log(beta) + t.digamma(alpha), alpha/(beta))
    @staticmethod
    def mean2conv(Elogx, Ex):
        """
        Generalised Newton's method from Eq. 10 in https://tminka.github.io/papers/minka-gamma.pdf
        Rewrite as:
        1/a^new = 1/a (1 + num / a (1/a + grad_digamma(a)))
        1/a^new = 1/a (1 + num / (1 + a grad_digamma(a)))
        a^new   = a / (1 + num / (1 + a grad_digamma(a)))
        """
        logEx = (Ex).log()
        diff = (Elogx - logEx)
        alpha = - 0.5 / diff
        for _ in range(6):
            num = diff + alpha.log() - t.digamma(alpha)
            denom = 1 - alpha * grad_digamma(alpha)
            alpha = alpha * t.reciprocal(1 + num/denom)
        beta = alpha / Ex
        return {'concentration': alpha, 'rate': beta}

    @staticmethod
    def test_conv(N):
        return {'concentration': t.randn(N).exp(), 'rate': t.randn(N).exp()}

    @staticmethod
    def canonical_conv(concentration, rate):
        return {'concentration': concentration, 'rate': rate}

#class InverseGammaMixin(AbstractMixin):
#PyTorch doesn't seem to have an Inverse Gamma distribution
#    dist = staticmethod(InverseGamma)
#    sufficient_stats = (t.log, t.reciprocal)
#
#    @staticmethod
#    def conv2nat(alpha, beta):
#        return (-alpha-1, -beta)
#    @staticmethod
#    def nat2conv(nat0, nat1):
#        return (-nat0-1, -nat1)
#
#    @staticmethod
#    def conv2mean(alpha, beta):
#        #From Wikipedia (Inverse Gamma: Properties)
#        return (t.log(beta) - t.digamma(alpha), alpha/beta)
#    @staticmethod
#    def mean2conv(mean_0, mean_1):
#        return GammaMixin.mean2conv(-mean_0, mean_1)
#
#    @staticmethod
#    def test_conv(N):
#        return (t.randn(N).exp(),t.randn(N).exp())

def vec_square(x):
    return x[..., :, None] @ x[..., None, :]
def posdef_matrix_inverse(x):
    return t.cholesky_inverse(t.linalg.cholesky(x))
def bmv(m, v):
    """
    Batched matrix vector multiplication.
    """
    return (m@v[..., None])[..., 0]
class MvNormalMixin(AbstractMixin):
    dist = staticmethod(MultivariateNormal)
    sufficient_stats = (identity,vec_square)
    # default_init_conv = {'loc': 0., 'covariance_matrix': t.eye(1)}
    @staticmethod
    def conv2nat(loc, covariance_matrix):
        P = posdef_matrix_inverse(covariance_matrix)
        return (bmv(P, loc), -0.5*P)
    @staticmethod
    def nat2conv(Pmu, minus_half_P):
        P = -2*minus_half_P
        S = posdef_matrix_inverse(P)
        return {'loc': bmv(S, Pmu), 'covariance_matrix': S}

    @staticmethod
    def conv2mean(loc, covariance_matrix):
        return (loc, covariance_matrix + vec_square(loc))
    @staticmethod
    def mean2conv(Ex, Ex2):
        return {'loc': Ex, 'covariance_matrix': Ex2 - vec_square(Ex)}

    @staticmethod
    def test_conv(N):
        mu = t.randn(N, 2)
        V = t.randn(N, 2, 4)
        S = V @ V.mT / 4
        return {'loc': mu, 'covariance_matrix': S}

    @staticmethod
    def canonical_conv(loc, covariance_matrix=None, precision_matrix=None, scale_tril=None):
        assert 1 == sum(x is not None for x in [covariance_matrix, precision_matrix, scale_tril])
        if precision_matrix is not None:
            covariance_matrix = posdef_matrix_inverse(precision_matrix)
        elif scale_tril is not None:
            covariance_matrix = scale_tril @ scale_tril.mT
        return {'loc': loc, 'covariance_matrix': covariance_matrix}

#    @staticmethod
#    def canonical_conv(loc, covariance_matrix=None, precision_matrix=None, scale_tril=None):
#        assert 1 == sum(x is not None for x in [covariance_matrix, precision_matrix, scale_tril])
#        if covariance_matrix is not None:
#            scale_tril = torch.linalg.cholesky(covariance_matrix)
#        elif precision_matrix is not None:
#            scale_tril = _precision_to_scale_tril(precision_matrix)
#        return {'loc': loc, 'scale_tril': scale_tril}
#
