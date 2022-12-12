import torch as t
from functorch.dim import Tensor

def is_dimtensor(tensor):
    return isinstance(tensor, Tensor)

def unify_dims(tensors):
    """
    Returns unique ordered list of dims for tensors in args
    """
    return ordered_unique([dim for tensor in tensors for dim in generic_dims(tensor)])

def generic_ndim(x):
    """
    Implements x.ndim, which is only defined for tensors
    """
    assert isinstance(x, (t.Tensor, Tensor, int, float))
    return x.ndim if isinstance(x, (t.Tensor, Tensor)) else 0

def generic_dims(x):
    """
    Implements x.dims, which is only defined for torchdim tensors
    """
    return x.dims if isinstance(x, Tensor) else ()

def generic_order(x, dims):
    """
    Implements x.order(dims), which is only defined for torchdim tensors
    """
    return x.order(*dims) if 0<len(dims) else x

def ordered_unique(ls):
    """
    Exploits the fact that in Python 3.7<, dict keys retain ordering

    Arguments:
        ls: list with duplicate elements
    Returns:
        list of unique elements, in the order they first appeared in ls
    """
    d = {l:None for l in ls}
    return list(d.keys())

def partition_tensors(lps, dim):
    """
    Partitions a list of tensors into two sets, one containing a given dim_name
    or only has interactions with tensors that have that dim name,
    one that doesn't
    """

    has_dim = [lp for lp in lps if dim     in set(generic_dims(lp))]
    no_dim  = [lp for lp in lps if dim not in set(generic_dims(lp))]

    return has_dim, no_dim

def max_dims(x, dims):
    #Ignore dims that aren't in the tensors
    set_xdims = set(generic_dims(x))
    dims = [dim for dim in dims if dim in set_xdims]

    if 0==len(dims):
        return x
    else:
        return generic_order(x, dims).flatten(0, len(dims)-1).max(0).values

def torchdim_einsum(tensors, sum_dims):
    #There shouldn't be any non-torchdim dimensions.
    #Should eventually be able to implement this as a straight product-sum
    for tensor in tensors:
        assert tensor.shape == ()

    set_sum_dims = set(sum_dims)

    all_dims = unify_dims(tensors)
    dim_to_idx = {dim: i for (i, dim) in enumerate(all_dims)}
    out_dims = [dim for dim in all_dims if dim not in set_sum_dims]
    out_idxs = [dim_to_idx[dim] for dim in out_dims]

    undim_tensors = []
    arg_idxs = []
    for tensor in tensors:
        dims = tensor.dims
        arg_idxs.append([dim_to_idx[dim] for dim in dims])
        undim_tensors.append(generic_order(tensor, dims))

    assert all(not is_dimtensor(tensor) for tensor in undim_tensors)

    einsum_args = [val for pair in zip(undim_tensors, arg_idxs) for val in pair] + [out_idxs]
    result = t.einsum(*einsum_args)
    if 0 < len(out_dims):
        result = result[out_dims]
    return result
        

def singleton_order(x, dims):
    """
    Takes a torchdim tensor and returns a standard tensor.

    x[dims] fails if any dims aren't present in x.
    This makes a new singleton dimension.
    """
    #Return immediately if not a dim tensor, as it broadcasts over everything
    if not is_dimtensor(x):
        return x

    #Ignore final Ellipsis
    if (len(dims) > 0) and (dims[-1] is Ellipsis):
        dims = dims[:-1]
    #No Ellipsis anywhere else
    assert Ellipsis not in dims

    x_dims = set(generic_dims(x))
    dims_present = [dim for dim in dims if dim in x_dims]
    idxs = [(slice(None) if (dim in x_dims) else None) for dim in dims]
    idxs.append(Ellipsis)

    result = generic_order(x, dims_present)[idxs]
    assert not is_dimtensor(result)
    return result


def dim2named_tensor(x, dims=None):
    """
    Doesn't need side information.
    Will fail if duplicated dim names passed in
    """
    if dims is None:
        dims = generic_dims(x)
    names = [repr(dim) for dim in dims]
    return generic_order(x, dims).rename(*names, ...)
