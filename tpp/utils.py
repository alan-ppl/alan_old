from functorch.dim import dims, Dim
import torch.nn as nn
from .tensor_utils import dename, dimtensormap, nameify



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
