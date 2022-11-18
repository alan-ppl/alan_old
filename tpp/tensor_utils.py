import torch
import torch.nn.functional as F
from functorch.dim import Tensor as DimTensor

def hasdim(x, lst):
  for element in lst:
    if  x is element:
      return True
  return False

def dename(tensors):

    assert isinstance(tensors, tuple)           or isinstance(tensors, list) \
        or isinstance(tensors, DimTensor) or isinstance(tensors, torch.Tensor)

    if isinstance(tensors, DimTensor):
        return tensors.order(*tensors.dims) if hasattr(tensors, 'dims') else tensors
    elif isinstance(tensors, torch.Tensor):
        return tensors
    else:
        return [tensor.order(*tensor.dims) if hasattr(tensor, 'dims') else tensor for tensor in tensors]

def expand(arg, shape):
    """
    Usual expand, but allows len(shape) != len(arg.shape), by expanding only the leading dimensions
    """
    return arg.expand(*shape, *arg.shape[-(len(arg.shape) - len(shape)):])


def typemap(f, typ, args, kwargs):
    """
    Apply a function to all args and kwargs of a given type
    """
    args = [(f(arg) if isinstance(arg, typ) else arg) for arg in args]
    kwargs = {key: (f(val) if isinstance(val, typ) else val) for (key, val) in kwargs.items()}
    return args, kwargs

def dimtensormap(f, args, kwargs):
    """
    Applys f to args and vals in kwargs if they are torch tensors
    """
    return typemap(f, DimTensor, args, kwargs)

def make_named(tensor):
    names = get_names(tensor)
    if names is None:
        return tensor
    else:
        return dename(tensor).refine_names(*names)

def tensormap(f, args, kwargs):
    """
    Applys f to args and vals in kwargs if they are torch tensors
    """
    return typemap(f, torch.Tensor, args, kwargs)


def tensors(args, kwargs):
    """
    Extract list of all tensors from args, kwargs
    """
    return [arg for arg in [*args, *kwargs.values()] if isinstance(arg, torch.Tensor)]


def pad_nones(arg, max_pos_dim):
    """
    Pad with as many positional dimensions as necessary after named dimensions
    to reach max_pos_dim
    """
    names = arg.names
    # current number of None's
    pos_dim = sum(name is None for name in names)
    # current number of named dimensions (excluding None's)
    named_dims = len(names) - pos_dim

    # strip names because unsqueeze can't cope with them
    arg = arg.rename(None)
    for _ in range(max_pos_dim-pos_dim):
        arg = arg.unsqueeze(named_dims)

    return arg.refine_names(*names, ...)

def get_names(tensor):
    dims = getattr(tensor, 'dims', None)
    if dims is None:
        return None
    else:
        names = [repr(dim) for dim in dims]
        return  names + [None]*(len(tensor.shape))

def get_dim_dict(tensors):
    dims = get_dims(tensors)
    return {repr(dim):dim for dim in dims}

def get_sizes(tensor):
    tensor = tensor.dims if hasattr(tensor, 'dims') else tensor
    return tuple([dim.size for dim in tensor])

def tensordim_to_name(tensors):
    named_tensors = []
    if not (isinstance(tensors, tuple) or isinstance(tensors, list)):
        tensors = [tensors]
    for tensor in tensors:
        if isinstance(tensors, DimTensor):
            names = get_names([tensor])
            denamed_tensor = dename([tensor])
            named_tensors.append(tensor.refine_names(*names))
    return named_tensors

def get_dim_dict(tensors):
    dims = get_dims(tensors)
    return {repr(dim):dim for dim in dims}

def get_dims(tensors):
    dims = []
    if not (isinstance(tensors, tuple) or isinstance(tensors, list)):
             tensors = [tensors]

    for tensor in tensors:
        if hasattr(tensor, 'dims'):
            for dim in getattr(tensor, 'dims', None):
                dims.append(dim)

    return dims
    
def sum_none_dims(lp):
    """
    Sum over None dims in lp
    """
    none_dims = [i for i in range(len(lp.names)) if lp.names[i] is None]
    if 0 != len(none_dims):
        lp = lp.sum(none_dims)
    return lp

def nameify(args, kwargs = {}):

    dim_dict = get_dim_dict(list(args) + list(kwargs.values()))
    args, kwargs = dimtensormap(lambda x: make_named(x), args, kwargs)
    def f(x, sample_dims=None, K_dim = None):
        names = x.names
        if sample_dims is not None:
            dim_dict[repr(sample_dims)] = sample_dims
        if K_dim is not None:
            dim_dict[repr(K_dim)] = K_dim

        if all([name is None for name in names]):
            return x
        else:
            dims = [dim_dict[name] if name is not None else None for name in names]
            while dims[-1] is None:
                dims.pop()
            dims = tuple(dims)
            return x.rename(None)[dims]

    return args, kwargs, f
