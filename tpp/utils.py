import torchdim
from torchdim import dims
import torch


def dename(tensors):

    assert isinstance(tensors, tuple)           or isinstance(tensors, list) \
        or isinstance(tensors, torchdim.Tensor) or isinstance(tensors, torch.Tensor)

    if isinstance(tensors, torchdim.Tensor):
        return tensors.order(*tensors.dims) if hasattr(tensors, 'dims') else tensors
    else:
        return [tensor.order(*tensor.dims) if hasattr(tensor, 'dims') else tensor for tensor in tensors]



def get_names(tensors):
    return [tensor.dims if hasattr(tensor, 'dims') else () for tensor in tensors]

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
