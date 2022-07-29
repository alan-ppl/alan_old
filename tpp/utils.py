import torchdim
from torchdim import dims
import torch
# from .cartesian_tensor import tensormap

def dename(tensors):

    assert isinstance(tensors, tuple)           or isinstance(tensors, list) \
        or isinstance(tensors, torchdim.Tensor) or isinstance(tensors, torch.Tensor)

    if isinstance(tensors, torchdim.Tensor):
        return tensors.order(*tensors.dims) if hasattr(tensors, 'dims') else tensors
    elif isinstance(tensors, torch.Tensor):
        return tensors
    else:
        return [tensor.order(*tensor.dims) if hasattr(tensor, 'dims') else tensor for tensor in tensors]


def hasdim(x, lst):
  for element in lst:
    if  x is element:
      return True
  return False


def is_K(dim_to_check):
    """
    Check that dim is correctly marked as 'K_'
    """
    return getattr(dim_to_check, 'K', False)

def is_plate(dim_to_check):
    """
    Check that dim is correctly marked as 'plate_'
    """
    return getattr(dim_to_check, 'plate', False)

def has_K(tensor):
    if hasattr(tensor, 'dims'):
        dims = tensor.dims
        return any([is_K(dim) for dim in dims])
    else:
        return False

def sum_K(tensor):
    if hasattr(tensor, 'dims'):
        dims = tensor.dims
        return sum([is_K(dim) for dim in dims])
    else:
        return 0

def has_plate(tensor):
    if hasattr(tensor, 'dims'):
        dims = tensor.dims
        return any([is_plate(dim) for dim in dims])
    else:
        return False

def sum_plate(tensor):
    if hasattr(tensor, 'dims'):
        dims = tensor.dims
        return sum([is_plate(dim) for dim in dims])
    else:
        return 0

def make_named(tensor):
    names = get_names(tensor)
    return dename(tensor).refine_names(*names)

def nameify(args, kwargs):
    dim_dict = get_dim_dict(list(args) + list(kwargs.values()))
    args, kwargs = tensormap(lambda x: make_named(x), args, kwargs)
    def f(x):
        names = get_names(x)
        dims = tuple([dim_dict[name] if name is not None else None for name in names])
        return x.rename(None)[dims]

    return args, kwargs, f


def get_dims(tensors):
    return [getattr(tensor, 'dims', None) for tensor in tensors if hasattr(tensor, 'dims')]

def get_names(tensor):
    dims = getattr(tensor, 'dims', None)
    if dims is None:
        return [None] * len(tensor.shape)
    else:
        return [repr(dim) for dim in dims]


def get_dim_dict(tensors):
    dims = get_dims(tensors)
    return {repr(dim):dim for dim in dims}

def get_sizes(tensor):
    tensor = tensor.dims if hasattr(tensor, 'dims') else tensor
    return tuple([dim.size for dim in tensor])

def make_K(dims):
    if not (isinstance(dims, tuple) or isinstance(dims, list)):
        dims = [dims]
    for dim in dims:
        if hasattr(dim, 'plate'):
            if getattr(dim, 'plate') == True:
                raise ValueError("Dims can't be both plate and K!")
        else:
            setattr(dim, 'K', True)
            setattr(dim, 'plate', False)

def make_plate(dims):
    if not (isinstance(dims, tuple) or isinstance(dims, list)):
        dims = [dims]
    for dim in dims:
        if hasattr(dim, 'K'):
            if getattr(dim, 'K') == True:
                raise ValueError("Dims can't be both plate and K!")
        else:
            setattr(dim, 'plate', True)
            setattr(dim, 'K', False)

def tensordim_to_name(tensors):
    named_tensors = []
    if not (isinstance(tensors, tuple) or isinstance(tensors, list)):
        tensors = [tensors]
    for tensor in tensors:
        if isinstance(tensors, torchdim.Tensor):
            names = get_names([tensor])
            denamed_tensor = dename([tensor])
            named_tensors.append(tensor.refine_names(*names))
    return named_tensors


def typemap(f, typ, args, kwargs):
    """
    Apply a function to all args and kwargs of a given type
    """
    args = [(f(arg) if isinstance(arg, typ) else arg) for arg in args]
    kwargs = {key: (f(val) if isinstance(val, typ) else val) for (key, val) in kwargs.items()}
    return args, kwargs

def tensormap(f, args, kwargs):
    """
    Applys f to args and vals in kwargs if they are torch tensors
    """
    return typemap(f, torch.Tensor, args, kwargs)


## Old Cartesian Tensor Stuff

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


def cartesiantensormap(f, args, kwargs):
    """
    Applys f to args and vals in kwargs if they are CartesianTensors
    """
    return typemap(f, CartesianTensor, args, kwargs)


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


def cartesian_tensorfy_value(val, unified_names):
    """
    Sometimes, pytorch operations return (nested) tuples of tensors.
    Go through tree, and convert all Tensors to CartesianTensors
    """
    if isinstance(val, torch.Tensor):
        return CartesianTensor(val.refine_names(*unified_names, ...))
    elif isinstance(val, tuple):
        return tuple(cartesian_tensorfy_value(v, unified_names) for v in val)
    else:
        return val


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
