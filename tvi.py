import math
import numpy as np
import torch as t
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.distributions import MultivariateNormal as MVN

# TVI
    # P
    # Q
    
# Not actually necessary, since trace takes over for it
class ParamNormal(nn.Module):
    def __init__(self, shape, mean=0, scale=1.):
        super().__init__()
        self.loc = nn.Parameter(t.ones(size=shape) * mean)
        self.log_scale = nn.Parameter(t.ones(shape) * math.log(scale))

        
    def forward(self):
        return Normal(self.loc, self.log_scale.exp())


class LinearNormal(nn.Module):
    def __init__(self, shape=t.Size([]), scale=1.):
        super().__init__()
        self.log_scale = nn.Parameter(t.ones(shape) * math.log(scale))

        
    def forward(self, input_):
        return Normal(input_, self.log_scale.exp())

