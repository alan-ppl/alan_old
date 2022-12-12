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
    has_dim = [lp for lp in lps if dim     in generic_dims(lp)]
    no_dim  = [lp for lp in lps if dim not in generic_dims(lp)]

    return has_dim, no_dim

def max_dims(x, dims):
    #Ignore dims that aren't in the tensors
    dims = list(set(generic_dims(tensor)).union(dims))

    if 0 == len(dims):
        return x
    else:
        return x.order(dims).flatten(0, len(dims)).max(0).values

def torchdim_einsum(tensors, sum_dims):
    #There shouldn't be any non-torchdim dimensions.
    #Should eventually be able to implement this as a straight product-sum
    for tensor in tensors:
        assert tensor.shape == ()

    all_dims = unify_dims(tensors)
    dim_to_idx = {dim: i for (i, dim) in enumerate(all_dims)}
    out_dims = [dim for dim in all_dims if dim not in sum_dims]
    out_idxs = [dim_to_idx[dim] for dim in out_dims]

    undim_tensors = []
    arg_idxs = []
    for tensor in tensors:
        dims = tensor.dims
        arg_idxs.append([dim_to_idx[dim] for dim in dims])
        undim_tensors.append(generic_order(tensor))

    einsum_args = [val for pair in zip(undim_tensors, arg_idxs) for val in pair] + [out_idxs]
    return t.einsum(*einsum_args)[out_dims]

def dim_align_to(x, dims):
    """
    Align to introduces singleton dimensions if they aren't present in x.
    x[dims] fails if any dims aren't present in x.
    This has the named tensor align_to behaviour, but works with dims.
    """
    x_dims = set(generic_dims(x))
    dims_present = [dim for dim in dims if dim in x_dims]
    idxs = [slice(None) if (dim in x_dims) else None for dim in dims]
    return generic_order(x, dims_present)[idxs]
