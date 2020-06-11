import re
import torch
import operator

def names(args):
    """
    ref is the reference list showing the cannonical ordering.
    """
    # get all the names in the NTensors
    names = []
    for arg in args:
        if isinstance(arg, NTensor):
            for name in arg.tensor.names:
                if name is not None:
                    names.append(name)
                    assert name in ref
    # convert to set to remove duplicates
    return sorted(set(names))

def broadcast(f):
    def inner(*args):
        return NTensor(f(*broadcast_args(*args)))
    return inner

def broadcast_args(*args):
    _names = names(args)
    _args = []
    for arg in args:
        if isinstance(arg, NTensor):
             _args.append(arg.tensor.clone().align_to(*_names, '...'))
        elif isinstance(arg, torch.Tensor):
             raise Exception("Encountered torch.Tensor, expected NTensor")
        else:
             _args.append(arg)
    return _args


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
        return NTensor(self.tensor.view(*self.tensor.shape_named, *args))
    def reshape(self, *args):
        return NTensor(self.tensor.reshape(*self.tensor.shape_named, *args))
    def transpose(self):
        raise NotImplementedError()


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
    we need this architecture (saving args and kwargs), because we need to broadcast together parameters and inputs
    """
    def __init__(self, dist, *args, **kwargs):
        self.dist = dist
        self.args = args
        self.kwargs = kwargs
    def rsample(self):
        return rsample(self.dist, *self.args, **self.kwargs)
    def log_prob(self, x):
        return log_prob(self.dist, x, *self.args, **self.kwargs)
    



nt = NT()

ref = ['c', 'b', 'a', 'z']
nt1 = NTensor(torch.ones(2,3,4,5), ("c", "b", "a"))
nt2 = NTensor(torch.ones(2,3,4,5), ("z", "b", "a"))

res = broadcast(operator.add)(nt1, nt2)


