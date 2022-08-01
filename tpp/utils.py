import torchdim
from torchdim import dims
import torch
from .cartesian_tensor import tensormap, typemap

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
    if names is None:
        return tensor
    else:
        return dename(tensor).refine_names(*names)

def nameify(args, kwargs):
    dim_dict = get_dim_dict(list(args) + list(kwargs.values()))
    args, kwargs = torchdimtensormap(lambda x: make_named(x), args, kwargs)
    def f(x, sample_dim=None, K_dim = None):
        names = x.names
        if sample_dim is not None:
            dim_dict[repr(sample_dim)] = sample_dim
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


def get_dims(tensors):
    return [getattr(tensor, 'dims', None) for tensor in tensors if hasattr(tensor, 'dims')]

def get_names(tensor):
    dims = getattr(tensor, 'dims', None)
    if dims is None:
        return None
    else:
        names = [repr(dim) for dim in dims]
        return  names + [None]*(len(tensor.shape) - len(names) + 1)


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

def torchdimtensormap(f, args, kwargs):
    """
    Applys f to args and vals in kwargs if they are torch tensors
    """
    return typemap(f, torchdim.Tensor, args, kwargs)

# def typemap(f, typ, args, kwargs):
#     """
#     Apply a function to all args and kwargs of a given type
#     """
#     args = [(f(arg) if isinstance(arg, typ) else arg) for arg in args]
#     kwargs = {key: (f(val) if isinstance(val, typ) else val) for (key, val) in kwargs.items()}
#     return args, kwargs
#
# def tensormap(f, args, kwargs):
#     """
#     Applys f to args and vals in kwargs if they are torch tensors
#     """
#     return typemap(f, torch.Tensor, args, kwargs)
