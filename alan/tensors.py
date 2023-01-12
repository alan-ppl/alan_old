import torch
import functorch.dim
from .pytree import treemap, values, Tensor

class AbstractAlanTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        # Use a Tensor that has the same metadata as wrapped_tensor for the
        # wrapper.
        return torch.Tensor._make_subclass(cls, torch.zeros((), device='meta'))

def check_args_kwargs(cls, args_kwargs):
    """
    Check that we only ever combine AlanTensors with other AlanTensors
    """
    for arg in values(args_kwargs):
        if isinstance(arg, functorch.dim.Tensor) or (isinstance(arg, torch.Tensor) and not isinstance(arg, AbstractAlanTensor)):
            raise Exception(f"Trying to combine {cls} and non-AlanTensor")

class TraceTensor(AbstractAlanTensor):
    pass

class NullTraceTensor(TraceTensor):
    """
    ```
    >>> NullTensor('a') + t.ones(3)
    NullTensor{'a'}
    >>> NullTensor('a') + NullTensor('b')
    NullTensor{'a', 'b'}
    ```
    Doesn't work with torchdim
    """
    @staticmethod
    def __new__(cls, keys):
        # Use a Tensor that has the same metadata as wrapped_tensor for the
        # wrapper.
        return torch.Tensor._make_subclass(cls, torch.zeros((), device='meta'))

    def __init__(self, keys):
        if isinstance(keys, str):
            keys = {keys}
        self.keys = set(keys)

    def __repr__(self):
        return f"NullTensor{self.keys}"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args_kwargs = (args, kwargs)
        check_args_kwargs(cls, args_kwargs)

        keys = set()
        for arg in values(args_kwargs):
            if isinstance(arg, NullTraceTensor):
                keys = keys.union(arg.keys)
        return NullTraceTensor(keys)

class ValuedTraceTensor(TraceTensor):
    """
    """
    def __init__(self, tensor):
        self.tensor = tensor

    def __repr__(self):
        return f"AlanTensor({self.tensor.__repr__()})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        args_kwargs = (args, kwargs)
        check_args_kwargs(cls, args_kwargs)
         
        for arg in values(args_kwargs):
            if isinstance(arg, NullTraceTensor):
                return NullTraceTensor.__torch_function__(func, types, args=args, kwargs=kwargs)

        args, kwargs = treemap(lambda x: (x.tensor if isinstance(x, AbstractAlanTensor) else x), args_kwargs)

        if torch.overrides.is_tensor_method_or_property(func) and isinstance(args[0], functorch.dim.Tensor):
            funcname = "functorch.dim." + torch.overrides.resolve_name(func)[6:]
            func = eval(funcname)
        results = func(*args, **kwargs)

        return treemap(lambda x: ValuedTraceTensor(x) if isinstance(x, Tensor) else x, results)

def unwrap_trace_tensor(x):
    if isinstance(x, Tensor):
        assert isinstance(x, ValuedTraceTensor)    
        return x.tensor
    else:
        return x

