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

