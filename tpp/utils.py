import torchdim
from torchdim import dims
import torch
from .prob_prog import TraceSample
from .tensor_utils import dename, torchdimtensormap, nameify

def make_dims(P, K, plates=None):
    tr = TraceSample()
    P(tr)
    names = list(tr.sample.keys())

    Ks = [torchdim.Dim(name='K', size=K)]
    for name in names:
        Ks.append(torchdim.Dim(name='K_{}'.format(name), size=K))


    # make_K(Ks)
    # if plates is not None:
    #     make_plate(plates)
    names = ['K'] + names
    dims = {names[i]:Ks[i] for i in range(len(names))}
    return dims


def hasdim(x, lst):
  for element in lst:
    if  x is element:
      return True
  return False


# def is_K(dim_to_check):
#     """
#     Check that dim is correctly marked as 'K_'
#     """
#     return getattr(dim_to_check, 'K', False)

# def is_plate(dim_to_check):
#     """
#     Check that dim is correctly marked as 'plate_'
#     """
#     return getattr(dim_to_check, 'plate', False)

# def has_K(tensor):
#     if hasattr(tensor, 'dims'):
#         dims = tensor.dims
#         return any([is_K(dim) for dim in dims])
#     else:
#         return False
#
# def sum_K(tensor):
#     if hasattr(tensor, 'dims'):
#         dims = tensor.dims
#         return sum([is_K(dim) for dim in dims])
#     else:
#         return 0

# def has_plate(tensor):
#     if hasattr(tensor, 'dims'):
#         dims = tensor.dims
#         return any([is_plate(dim) for dim in dims])
#     else:
#         return False
#
# def sum_plate(tensor):
#     if hasattr(tensor, 'dims'):
#         dims = tensor.dims
#         return sum([is_plate(dim) for dim in dims])
#     else:
#         return 0


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
