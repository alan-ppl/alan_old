from functorch.dim import dims, Dim
import torch.nn as nn
from .prob_prog import TraceSample
from .tensor_utils import dename, dimtensormap, nameify

def make_dims(P, K, plates=None):
    tr = TraceSample()
    P(tr)
    names = list(tr.sample.keys())

    Ks = [Dim(name='K', size=K)]
    for name in names:
        Ks.append(Dim(name='K_{}'.format(name), size=K))


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

class Q_module(nn.Module):
    def __init__(self):
        super().__init__()

    def reg_param(self, name, tensor, dims=None):
        parameter_name = "_"+name
        self.register_parameter(parameter_name, nn.Parameter(tensor)) #nn.Parameter?
        def inner():
            tensor = getattr(self, parameter_name)
            if dims is None:
                return tensor
            else:
                return tensor[dims]
        setattr(self, name, inner())
