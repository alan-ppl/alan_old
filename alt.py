import torch as t
import torch.distributions as td
import torch.nn as nn

#### Wrapping dists so that they propagate named dimensions correctly

def unify_names(*nss):
    return sum(t.zeros(len(ns)*(0,), names=ns) for ns in nss).names

def unify_arg_names(*args):
    return unify_names(*(arg.names for arg in args if isinstance(arg, t.Tensor)))

def strip_name(arg):
    if isinstance(arg, t.Tensor):
        return arg.rename(None)
    else:
        return arg

def strip_names(*args):
    return (strip_name(arg) for arg in args)


class WrappedDist:
    def __init__(self, dist, *args, sample_shape=(), sample_names=()):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        if isinstance(sample_names, str):
            sample_names = (sample_names,)

        if sample_names==() and sample_shape!=():
            sample_names = len(sample_shape) * (None,)

        assert any(isinstance(arg, t.Tensor) for arg in args)

        self.dist = dist(*strip_names(*args))
        self.unified_names = unify_arg_names(*args)
        self.sample_shape = sample_shape
        self.sample_names = sample_names
    
    def rsample(self):
        return self.dist.rsample(sample_shape=self.sample_shape) \
                .refine_names(*self.sample_names, *self.unified_names)
    
    def log_prob(self, x):
        return self.dist.log_prob(x) \
                .refine_names(*unify_names(x.names, self.unified_names))




class Trace:
    def __init__(self, K_shape, K_names):
        """
        Initialize all Trace objects with K_shape and K_names.
        These should form the rightmost dimensions in all tensors in the program.
        """
        assert len(K_shape) == len(K_names)
        self.K_shape = K_shape
        self.K_names = K_names

    def pad(self, tensor):
        """
        Pad an external tensor with the current K_name and K_shape
        """
        tensor = tensor.align_to(..., *self.K_names)
        tensor = tensor.expand((*tensor.shape[:(len(tensor.shape)-len(self.K_shape))], *self.K_shape))
        return tensor

    def compatible_tensor(self, tensor):
        """
        Check that a tensor is compatible with the current K_name and K_shape
        """
        assert self.K_names == tensor.names[-len(self.K_names):]
        assert self.K_shape == tensor.shape[-len(self.K_shape):]
    
    def compatible_arg(self, arg):
        """
        If arg is a tensor, check that it is compatible with the current K_name and K_shape
        """
        if isinstance(arg, t.Tensor):
            self.compatible_tensor(arg)

    def compatible_args(self, *args, **kwargs):
        """
        If any arg or kwarg is a tensor, check that it is compatible with the current K_name and K_shape
        """
        args = (*args, *kwargs.values())
        for arg in args:
            self.compatible_arg(arg)

    def dist(self, dist, *args, **kwargs):
        """
        Check arg and kwarg are compatible before passing them to WrappedDist
        """
        self.compatible_args(*args, **kwargs)
        return WrappedDist(dist, *args, **kwargs)


#### Add padded tensor initialization ops to Trace
init_names = [
    "tensor",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "eye",
    "range",
    "linspace",
    "logspace",
    "empty",
    "empty_like",
    "full",
    "full_like",
    "complex",
    "polar",
]

def init_name_to_func(init_name):
    torch_func = getattr(t, init_name)
    return lambda self, *args, **kwargs: self.pad(torch_func(*args, **kwargs))
init_funcs = [init_name_to_func(init_name) for init_name in init_names]
for init_name, init_func in zip(init_names, init_funcs):
    setattr(Trace, init_name, init_func)
        

dist_names = [
    "Bernoulli",
    "Beta",
    "Binomial",
    "Categorical",
    "Cauchy",
    "Chi2",
    "ContinuousBernoulli",
    "Exponential",
    "FisherSnedecor",
    "Gamma",
    "Geometric",
    "Gumbel",
    "HalfCauchy",
    "HalfNormal",
    "Laplace",
    "LogNormal",
    "NegativeBinomial",
    "Normal",
    "Pareto",
    "Poisson",
    "RelaxedBernoulli",
    "StudentT",
    "Uniform",
    "VonMises",
    "Weibull",
]

#Multivariate!
#Dirichlet
#MvNormal
#Multinomial
#OneHotCategorical
#RelaxedOneHotCategorical

def set_dist(dist_name):
    dist = getattr(td, dist_name)
    def inner(*args, **kwargs):
        return WrappedDist(dist, *args, **kwargs)
    globals()[dist_name] = inner

for dist_name in dist_names:
    set_dist(dist_name)


class TraceSampleLogP(Trace):
    """
    Samples a probabilistic program + evaluates log-probability.
    Usually used for sampling the approximate posterior.
    Doesn't do any tensorisation.  All dimensions (e.g. sampling K) is managed by the user's program.
    Note that the latents may depend on the data (as in a VAE), but it doesn't make sense to "sample" data.
    """
    def __init__(self, K, data=None):
        super().__init__(K_shape=(K,), K_names=("K",))
        if data is None:
            data = {}
        self.data = data
        self.sample = {}
        self.logp = {}
    
    def __getitem__(self, key):
        if key in self.sample:
            assert key not in self.data
            return self.sample[key]
        else:
            assert key in self.data
            return self.data[key]

    def __setitem__(self, key, value):
        assert isinstance(value, WrappedDist)
        assert key not in self.data
        assert key not in self.sample
        sample = value.rsample()
        self.sample[key] = sample
        self.logp[key] = value.log_prob(sample)

class TraceSample(Trace):
    """
    Just used e.g. to test a program or to sample from the prior, and as such it doesn't make sense for it to take data as input.
    No K's.
    """
    def __init__(self):
        super().__init__(K_shape=(), K_names=())
        self.sample = {}
    
    def __getitem__(self, key):
        return self.sample[key]

    def __setitem__(self, key, value):
        assert isinstance(value, WrappedDist)
        assert key not in self.sample
        self.sample[key] = value.rsample()
    
    def set_names(*args, **kwargs):
        pass
    
    def add_remove_names(*args, **kwargs):
        pass

class TensorisedTrace(Trace):
    def __init__(self, sample, data):
        self.sample = sample
        self.data = data
        #maintains an ordered list of tensors as they are generated
        self.logp = {}
        self.K_names = [f"K_{name}" for name in sample.keys()]
        self.K_shape = [1 for name in sample.keys()]

    def set_names(self, *names):
        self.K_names = [f"K_{name}" for name in names]
        self.K_shape = [1 for name in names]

    def add_remove_names(self, add_names, remove_names, tensors=()):
        """
        Updates K_shape and K_names in TensorisedTrace.
        This automatically updates names in sample a they are pulled out, so we don't have to do anything to them explicitly.
        We also give a list of tensors, in case there are any that need dimensions updating.
        Allows us to work around limitations in the number of dimensions baked into PyTorch.
        """
        #remove names from K_names and K_shape
        #intentionally causes error if name isn't present.
        for name in remove_names:
            self.K_names.remove(f"K_{name}")
            self.K_shape.pop()

        # remove names from tensors 
        # deliberately errors if name isn't present, or if dimension shape is more than one
        tensors = [squeeze_dims(tensor, names) for tensor in tensors]

        for name in add_names:
            self.K_names.append(f"K_{name}")
            self.K_shape.append(1)
            tensors = [tensor.unsqueeze(-1).refine_names(..., name) for tensor in tensors]

        return tensors


    def __getitem__(self, key):
        #ensure tensor has been generated
        assert (key in self.data) or (key in self.sample)
        
        if key in self.sample:
            sample = self.sample[key].rename(K=f"K_{key}")
        else:
            sample = self.data[key]

        sample = sample.align_to(..., *self.K_names)
        return sample
            

    def __setitem__(self, key, value):
        assert isinstance(value, WrappedDist)
        assert (key in self.data) or (key in self.sample)
        sample = self[key]
        self.logp[key] = value.log_prob(sample)



#### Example

def P(tr): 
    tr.set_names('a', 'b')
    tr['a'] = Normal(tr.zeros(()), 1)
    tr['b'] = Normal(tr['a'], 1)
    tr.add_remove_names(('c',), ('a',))
    tr['c'] = Normal(tr['b'], 1, sample_shape=3, sample_names='plate_a')
    print(tr['c'].names)
    print(tr['c'].shape)
    tr['obs'] = Normal(tr['c'], 1, sample_shape=5, sample_names='plate_b')


class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_a = nn.Parameter(t.zeros(()))
        self.m_b = nn.Parameter(t.zeros(()))
        self.m_c = nn.Parameter(t.zeros((3,), names=('plate_a',)))

    def forward(self, tr):
        tr['a'] = Normal(tr.pad(self.m_a), 1)
        tr['b'] = Normal(tr.pad(self.m_b), 1)
        tr['c'] = Normal(tr.pad(self.m_c), 1)



#sample fake data
tr_sample = TraceSample()
P(tr_sample)
data = {'obs': tr_sample.sample['obs']}

#sample from approximate posterior
trq = TraceSampleLogP(K=10, data=data)
q = Q()
q(trq)

#
#compute logP
trp = TensorisedTrace(trq.sample, data)
P(trp)

