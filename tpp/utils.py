from functorch.dim import dims, Dim
import torch.nn as nn
from .prob_prog import TraceSample
from .tensor_utils import dename, dimtensormap, nameify

def make_dims(P, K):
    tr = TraceSample()
    P(tr)

    groups = {}
    for v in set(tr.groups.values()):
        groups[v] = Dim(name='K_{}'.format(v), size=K)
    dims = {'K':Dim(name='K', size=K)}
    for k,v in tr.groups.items():
        dims[k] = groups[v] if tr.groups[k] is not None else Dim(name='K_{}'.format(k), size=K)

    return dims


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
