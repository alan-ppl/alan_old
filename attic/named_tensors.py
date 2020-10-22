import re
import torch
import operator

def names(*args, **kwargs):
    """
    extracts all the names in args and kwargs, and lists them in alphabetical order.
    """
    names = []
    pos = 0
    for arg in [*args, *kwargs.values()]:
        if isinstance(arg, NTensor):
            pos = max(pos, arg.positional)

            for name in arg.tensor.names:
                if name is not None:
                    names.append(name)
                    assert name in ref

        elif isinstance(arg, torch.Tensor):
            pos = max(pos, len(arg.shape))

            # Any torch.Tensors are unnamed
            assert arg.names == len(arg.shape)*(None,)
    # convert to set to remove duplicates, sort, and pad with Nones for positionals nones
    return (*sorted(set(names)), *(pos*(None,)))


def nones(ls):
    return sum(l is None for l in ls)


def broadcast_arg(arg, names):
    if isinstance(arg, NTensor):
        # The goal is to get tensor.names == names.  
        # But this is harder than should be because align_to doesn't understand Nones...
        tensor = arg.tensor

        # positional args in names, tensor and the difference
        none_names = nones(names)
        none_tensor = nones(tensor.names)
        none_diff = none_names - none_tensor

        #align tensor, but with the wrong number of positional args
        not_none_names = names[:-none_names]
        tensor = arg.tensor.clone().align_to(*not_none_names, '...')

        # view doesn't work yet with named tensors, so drop named args
        tensor = tensor.rename(None)
        # introduce singleton dims
        tensor = tensor.view((*tensor.shape[:-none_tensor], 
                              *(none_diff * (1,)), 
                              *tensor.shape[-none_tensor:]))
        
        tensor = tensor.refine_names(*names)
        return tensor
    else:
        # leave torch.Tensor and other types
        return arg


def broadcast_args(*args, **kwargs):
    _names = names(*args, **kwargs)
    args = [broadcast_arg(arg, _names) for arg in args]
    kwargs = {k: broadcast_arg(v, names) for (k, v) in kwargs.items()}
    return args, kwargs

def broadcast(f):
    def inner(*args, **kwargs):
        args, kwargs = broadcast_args(*args, **kwargs)
        return NTensor(f(*args, **kwargs))
    return inner



class NTensor:
    """
    thin wrapper on a PyTorch tensor
    must use negative axis indexes!
    
    """
    def __init__(self, tensor, names=None):
        assert isinstance(tensor, torch.Tensor)
        self.tensor = tensor
        if names is not None:
            self.tensor = self.tensor.refine_names(*names, ...)

    @property
    def shape(self):
        return self.tensor.shape[self.named:]

    @property
    def shape_named(self):
        return self.tensor.shape[:self.named]

    @property
    def named(self):
        return sum(name is not None for name in self.tensor.names)

    @property
    def positional(self):
        return sum(name is     None for name in self.tensor.names)

    def view(self, *args):
        return NTensor(self.tensor.view(*self.shape_named, *args))
    def reshape(self, *args):
        return NTensor(self.tensor.reshape(*self.shape_named, *args))
    def transpose(self):
        raise NotImplementedError()

    _sum = broadcast(torch.sum)
    def sum(self, dim, keepdim=False, dtype=None):
        return self._sum(dim, keepdim=keepdim, dtype=None)


    __add__ = broadcast(operator.add)
    __radd__ = broadcast(operator.add)

    __sub__ = broadcast(operator.sub)
    __rsub__ = broadcast(operator.sub)

    __mul__ = broadcast(operator.mul)
    __rmul__ = broadcast(operator.mul)

    __floordiv__ = broadcast(operator.floordiv)
    __rfloordiv__ = broadcast(operator.floordiv)

    __truediv__ = broadcast(operator.truediv)
    __rtruediv__ = broadcast(operator.truediv)

    __mod__ = broadcast(operator.mod)
    __rmod__ = broadcast(operator.mod)

    __pow__ = broadcast(operator.pow)
    __rpow__ = broadcast(operator.pow)

    __matmul__ = broadcast(operator.matmul)
    __rmatmul__ = broadcast(operator.matmul)

    __pos__ = broadcast(operator.pos)
    __neg__ = broadcast(operator.neg)

    def __getattr__(self, attr):
        def inner(*args, **kwargs):
            return broadcast(getattr(torch, attr))(self, *args, **kwargs)
        return inner



class NT:
    """
    Use in place of torch library.
    Intercepts calls and wraps them in broadcast
    """
    def __getattr__(self, attr):
        return broadcast(getattr(torch, attr))


#### Wrapped distributions

class NDistributions:
    """
    Use in place of torch library.
    Intercepts distributions and wraps them in NDist
    """
    def __getattr__(self, attr):
        uppercase = re.match(r'^[A-Z]', attr)
        lowercase = re.match(r'^[a-z]', attr)
        assert uppercase or lowercase

        def _NDist(*args, **kwargs):
            return NDist(getattr(torch.distributions, attr.capitalize()), *args, **kwargs)
        
        if uppercase:
            return _NDist
        else:
            def inner(trace, *args, K=None, group=None, **kwargs):
                return trace.primitive(_NDist(*args, **kwargs), K=K, group=group)
            return inner


ndistributions = NDistributions()

#x, *args and **kwargs are tensors, so all can be broadcast
def _rsample(dist, *args, **kwargs):
    """
    broadcastable rsample, that takes the distribution's args and kwargs as Tensor arguments
    """
    return dist(*args, **kwargs).rsample()

rsample = broadcast(_rsample)

def _log_prob(dist, x, *args, **kwargs):
    """
    broadcastable log-prob, that takes the the location, x, and the distribution's args and kwargs as Tensor arguments
    """
    return dist(*args, **kwargs).log_prob(x)

log_prob = broadcast(_log_prob)


class NDist:
    """
    Wrapper for primitive distributions
    saves args and kwargs as NTensors (because we need to broadcast together parameters and inputs)
    """
    def __init__(self, dist, *args, **kwargs):
        self.dist = dist
        self.args = args
        self.kwargs = kwargs
        
    def rsample(self):
        return rsample(self.dist, *self.args, **self.kwargs)
    
    def log_prob(self, x):
        return log_prob(self.dist, x, *self.args, **self.kwargs)


class Nnn():
    """
    Use in place of torch.nn library
    __getattr__ is a factor function that creates a new class, inheriting from e.g. nn.Linear,
    and overrides __call__
    """
    def __getattr__(self, attr):
        class Inner(getattr(torch.nn, attr)):
            def __call__(self, *args, **kwargs):
                return broadcast(super().__call__)(*args, **kwargs)
            #also wrap repr
        return Inner

nnn = Nnn()



nt = NT()

ref = ['c', 'b', 'a', 'z']
nt1 = NTensor(torch.ones(2,3,4,5,6), ("c", "b", "a"))
nt2 = NTensor(torch.ones(2,3,4,6), ("z", "b", "a"))

res = broadcast(operator.add)(nt1, nt2)
