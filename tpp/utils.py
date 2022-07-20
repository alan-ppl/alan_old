import torchdim
import torch

def has_K(tensor):
    if hasattr(tensor, 'dims'):
        return 'K' in list(tensor.dims)
    else:
        return False

def dename(tensors):

    assert isinstance(tensors, tuple)           or isinstance(tensors, list) \
        or isinstance(tensors, torchdim.Tensor) or isinstance(tensors, torch.Tensor)

    if isinstance(tensors, torchdim.Tensor):
        return tensors.order(*tensors.dims) if hasattr(tensors, 'dims') else tensors
    else:
        return [tensor.order(*tensor.dims) if hasattr(tensor, 'dims') else tensor for tensor in tensors]



# def dename(tensors, K):
#     print(tensors)
#     print(K)
#     if hasattr(tensors[0], 'dims'):
#         print('K' in list(tensors[0].dims))
#         print('K' in tensors[0].dims)
#     if K:
#         return [tensor.order(tensor.dims) if hasattr(tensor, 'dims') else tensor for tensor in tensors]
#     else:
#         return [tensor.order(tensor.dims).reshape(tuple(i.size for i in tensor.dims)) if hasattr(tensor, 'dims') else tensor for tensor in tensors]



def get_names(tensors):
    return [tensor.dims if hasattr(tensor, 'dims') else () for tensor in tensors]

def get_sizes(tensor):
    tensor = tensor.dims if hasattr(tensor, 'dims') else tensor
    return tuple([dim.size for dim in tensor])
