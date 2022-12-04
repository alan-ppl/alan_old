import torch as t
    
class ScaleTensor:
    """
    Represents a signed tensor with a wide range of orders of magnitude as a sign + a log-scale
    """
    def __init__(self, x=1., log_scale=0.):
        """
        x and log_scale can be any tensor (but they must broadcast together).
        The resulting tensor is: x * log_scale.exp()
        We can convert a standard tensor by giving only an x argument.
        We can convert a log-probability tensor by giving only a log_scale argument.
        """
        assert isinstance(x, t.Tensor) or isinstance(log_scale, t.Tensor)

        self.sign = t.sign(x)
        #sign_x is zero or positive
        sign_x = x*self.sign 
        #pos_x is always positive (its value doesn't matter when sign=0, as we multiply by sign).
        pos_x = sign_x + (self.sign==0).to(dtype=sign_x.dtype)
        self.log_scale = log_scale + pos_x.log()

    @property
    def names(self):
        return self.log_scale.names

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
        elif 1 == len(dims):
            max_ = self.log_scale.max(dims[0]).values
        else:
            assert all(name in self.names for name in dims)
            ordered_dims = tuple(name for name in self.log_scale.names if name in dims)
            max_ = self.log_scale.flatten(ordered_dims, 'flattened').max('flattened').values

        return self.sign * (self.log_scale - max_.align_as(self.log_scale)).exp(), max_

    def sum(self, dim):
        x, log_scale = self.normalized([dim])
        return ScaleTensor(x.sum(dim), log_scale)

if __name__ == "__main__":
    x = t.randn(3,3, names=("a", "b"))
    xs = x.sum("a")
    st_x = ScaleTensor(x)
    print(xs)
    st_xs = st_x.sum("a")
    print(st_xs.tensor())
