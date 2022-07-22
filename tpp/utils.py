import torchdim
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
