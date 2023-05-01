import torch as t
import torch.nn as nn
from .dist import *
from .utils import *
from .alan_module import AlanModule
from .exp_fam_mixin import *

class GLM(AlanModule):
    """
    Generalised linear models

    GLMs
    Only use cannonical link (otherwise you need to map from mean to natural params, and differentiate back through that mapping).
    Only really makes sense for Linear and Logistic Regression (and maybe Poisson).
    Combine inputs (which the dist holds on to) with gradients of beta.
     (which the dist holds on to)
    """
    def __init__(self, platesizes=None, sample_shape=(), init_beta=None):
        super().__init__()
        # if init_beta is None:
        #     init_beta = 0

        # self.inputs = inputs

        if platesizes is None:
            platesizes = {}
        shape = [*platesizes.values(), *sample_shape]
        names = [*platesizes.keys(), *(len(sample_shape) * [None])]

        self.betanames = tuple(f'beta_{i}' for i in range(len(self.sufficient_stats)))
        for betaname in self.betanames:
            self.register_parameter(betaname, nn.Parameter(t.zeros(shape).rename(*names)))

        # self.register_buffer('beta', t.full(shape, init_beta).rename(*names))



        self.platenames = tuple(platesizes.keys())

    @property
    def dim_betas(self):
        return [getattr(self, betaname) for betaname in self.betanames]
    @property
    def named_betas(self):
        return [self.get_named_tensor(betaname) for betaname in self.betanames]

    @property
    def named_grads(self):
        return [self.get_named_grad(betaname) for betaname in self.betanames]


    def _update(self, lr):
        """
        Newtons update?
            1. Score function
            2. Negative Hessian
            3. Perform update?
        Or just gradient descent?
        """
        for (beta, grad) in zip(self.named_betas, self.named_grads):
            beta.data.add_(grad, alpha=lr)

    def __call__(self, inputs):
        mean = []
        for beta in self.dim_betas:
            mean.append(inputs * beta)
        return self.dist(**self.canonical_conv(*mean))

    def local_parameters(self):
        return self.beta


class LinearRegression(GLM, NormalMixin):
    pass
class LogisticRegression(GLM, BernoulliMixin):
    pass
class PoissonRegression(GLM, PoissonMixin):
    pass
