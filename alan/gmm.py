import torch as t
import torch.nn as nn
from .dist import *
from .utils import *
from .alan_module import AlanModule
from .exp_fam_mixin import *

class GMM(AlanModule):
    """
    Gaussian mixture models

    TODO:
    Mixture models
      Most of the obvious approaches won't work too well due to symmetry (e.g. each datapoint has an equal chance of being assigned to each cluster).
      The right approach is to define a prior + approximate posterior over partitions.
      Approximate posterior should be written in terms of "affinities" (datapoint 4 wants to be with datapoint 3 but not datapoints 1 or 2).
      We can use the partition to define a dynamic plate (as we have independence across clusters).
    """
    def __init__(self, num_partitions, platesizes=None, sample_shape=()):
        self.dist = staticmethod(Normal)


        self.num_partitions = num_partitions



        if platesizes is None:
            platesizes = {}

        shape = [*platesizes.values(), *sample_shape]
        names = [*platesizes.keys(), *(len(sample_shape) * [None])]

        ## Need to find last platedim size?

        self.mean_names = tuple(f'mean_{i}' for i in range(num_partitions))
        self.sigma_names = tuple(f'sigma_{i}' for i in range(num_partitions))
        for i in range(num_partitions):
            # self.partition_sizes['part_{}'.format(i)] = size//num_partitions}
            self.register_parameter('mean_{}'.format(i), nn.Parameter(t.zeros(shape).rename(*names)))
            self.register_parameter('sigma_{}'.format(i),  nn.Parameter(t.zeros(shape).rename(*names)))

        # self.partition_sizes['part_{}'.format(num_partitions-1)] = size//num_partitions + size % num_partitions

        self.register_parameter('pi', nn.Parameter((1/num_partitions) * t.ones(num_partitions, shape).rename(*names)))

    @property
    def dim_betas(self):
        return [getattr(self, betaname) for betaname in self.betanames]
    @property
    def named_betas(self):
        return [self.get_named_tensor(betaname) for betaname in self.betanames]

    @property
    def named_grads(self):
        return [self.get_named_grad(betaname) for betaname in self.betanames]

    @property
    def pi_prior(self):
        # return Dirichlet(self.pi)
        return Categorical(self.pi)

    def
