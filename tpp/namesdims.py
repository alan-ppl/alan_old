from functorch.dim import Dim
from .utils import generic_ndim, generic_dims, generic_order

class NamesDims():
    """
    A two way dict name (str) <-> dim (functorch.dim.Dim)
    Functional API
    """
    def __init__(self, name2dim=None, dim2name=None):
        if name2dim is None:
            name2dim = {}
        if dim2name is None:
            dim2name = {}
        self.name2dim = name2dim
        self.dim2name = dim2name

    def insert_size(self, name, size):
        if name in self.name2dim:
            assert size == name2dim[name].size
            return self
        else:
            dim = Dim(name, size)
            return NamesDims({name: dim, **self.name2dim},{dim: name, **self.dim2name})

    def insert_size_dict(self, d):
        for (name, size) in d.items():
            self = self.insert_size(name, size)
        return self

    def insert_named_tensor(self, d):
        for (name, size) in zip(d.names, d.shape):
            if name is not None:
                self = self.insert_size(name, size)
        return self

    def cat(self, other):
        name2dim = {**self.name2dim, **other.name2dim}
        dim2name = {**self.name2dim, **other.name2dim}
        assert len(name2dim) == len(self.name2dim) + len(other.dim2name)
        assert len(dim2name) == len(self.dim2name) + len(other.dim2name)
        return NamesDims(name2dim, dim2name)

    def named2dim_tensor(self, x):
        torchdims = [slice(None) if (name is None) else self.name2dim[name] for name in x.names]
        return x.rename(None)[torchdims]

    def dim2named_tensor(self, x):
        dims = generic_dims(x)
        names = [self.dim2name[dim] for dim in dims]
        return generic_order(x, dims).rename(*names, ...)

