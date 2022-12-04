import torch as t
    
class ScaleTensor:
    """
    Represents a signed tensor with a wide range of orders of magnitude as a sign + a log-scale
    """
    def __init__(self, x=1., log_scale=0.):
        """
        x and log_scale can be any signed tensor.
        The resulting tensor is: x * log_scale.exp()
        """
        self.sign = t.sign(x)
        #x is always positive, even if zero.
        x = x*self.sign + self.sign==0
        self.log_scale = log_scale + x.log()

    def tensor(self):
        """
        Returns a standard tensor
        """
        return self.sign * self.log_scale.exp()

    def normalized(self, dims):
        """
        Returns a standard tensor
        """
        if 0 == len(dims):
            max_ = self.log_scale
        else:
            ordered_dims = tuple(name for name in self.log_scale.names if name in dims)
            max_ = self.log_scale.flatten(ordered_dims, 'flattened').max('flattened').values

        return self.sign * (self.log_scale - max_.align_as(self.log_scale)).exp(), max_

    def sum(self, dim):
        x, log_scale = normalized(self, [dim])
        return ScaleTensor(x.sum(dim), log_scale)
