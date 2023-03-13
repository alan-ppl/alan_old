import math
import torch as t
import functorch
from functorch.dim import Dim

Tensor = (functorch.dim.Tensor, t.Tensor)
Number = (int, float)
TensorNumber = (*Tensor, *Number)

#### Utilities for working with torchdims
def sum_non_dim(x):
    """
    Sums over all non-torchdim dimensions.
    Returns x for anything that isn't a tensor.
    """
    return x.sum() if (isinstance(x, Tensor) and x.ndim > 0) else x

"""
Defines a series of reduction functions that are called e.g. as
sum_dims(x, (i, j)), where i, j are torchdims.
"""
def assert_iter(dims, varname='dims'):
    if not isinstance(dims, (list, tuple)):
        raise Exception(varname + ' must be a list or tuple')

def assert_unique_iter(dims, varname='dims'):
    assert_iter(dims, varname=varname)
    if len(set(dims)) != len(dims):
        raise Exception(f'Non-unique elements in {varname}')

def assert_unique_dim_iter(dims, varname='dims'):
    assert_unique_iter(dims, varname=varname)
    for dim in dims:
        if not isinstance(dim, Dim):
            raise Exception(f'dim in {varname} is not torchdim dimension')

def assert_no_ellipsis(dims):
    if 0<len(dims):
        assert dims[-1] is not Ellipsis

def reduce_dims(func):
    """
    Reduces over specified torchdim dimensions.
    Returns itself if no dims given.
    """
    def inner(x, dims):
        assert_unique_dim_iter(dims)
        if 0<len(dims):
            if not isinstance(x, functorch.dim.Tensor):
                raise Exception("dims provided, but x is not a torchdim tensor.")
            x = getattr(x.order(dims), func)(0)
        return x
    return inner

sum_dims        = reduce_dims("sum")
prod_dims       = reduce_dims("prod")
mean_dims       = reduce_dims("mean")
min_dims        = reduce_dims("min")
max_dims        = reduce_dims("max")
logsumexp_dims  = reduce_dims("logsumexp")

def logmeanexp_dims(x, dims):
    return logsumexp_dims(x, dims) - sum([math.log(dim.size) for dim in dims])


def is_dimtensor(tensor):
    return isinstance(tensor, functorch.dim.Tensor)

def unify_dims(tensors):
    """
    Returns unique ordered list of dims for tensors in args
    """
    return ordered_unique([dim for tensor in tensors for dim in generic_dims(tensor)])

def generic_ndim(x):
    """
    Generalises x.ndim, which is only defined for tensors
    """
    assert isinstance(x, TensorNumber)
    return x.ndim if isinstance(x, Tensor) else 0

def generic_dims(x):
    """
    Generalises x.dims, which is only defined for torchdim tensors
    """
    return x.dims if is_dimtensor(x) else ()

def generic_order(x, dims):
    """
    Generalises x.order(dims), which is only defined for torchdim tensors
    """
    assert_unique_dim_iter(dims)
    assert_no_ellipsis(dims)

    #If x is not a dimtensor, then we can't have any dims.
    if not is_dimtensor(x):
        assert 0 == len(dims)

    return x.order(*dims) if 0<len(dims) else x

def generic_getitem(x, dims):
    assert_iter(dims) #dims doesn't have to be unique, e.g. [2,2]
    assert_no_ellipsis(dims)

    if len(dims)==0:
        return x
    else:
        return x[dims]

def generic_setitem(x, dims, value):
    assert_iter(dims) #dims doesn't have to be unique, e.g. [2,2]
    assert_no_ellipsis(dims)

    if len(dims)==0:
        dims = (Ellipsis,)

    x[dims] = value

def ordered_unique(ls):
    """
    Exploits the fact that in Python 3.7<, dict keys retain ordering

    Arguments:
        ls: list with duplicate elements
    Returns:
        list of unique elements, in the order they first appeared in ls
    """
    assert_iter(ls, 'ls')
    d = {l:None for l in ls}
    return list(d.keys())

def partition_tensors(lps, dim):
    """
    Partitions a list of tensors into two sets, one list with all tensors
    that have dim, and another list with all tensors that don't have that
    dim
    """
    has_dim = [lp for lp in lps if dim     in set(generic_dims(lp))]
    no_dim  = [lp for lp in lps if dim not in set(generic_dims(lp))]

    return has_dim, no_dim


def singleton_order(x, dims):
    """
    Takes a torchdim tensor and returns a standard tensor,
    in a manner that mirrors `x.order(dims)` (if `dims` had all
    dimensions in `x`). However, with `x.order(dims)`,
    all dimensions in `dims` must be in `x`.  Here, we allow new
    dimensions in `dims` (i.e. dimensions in `dims` that aren't
    in `x`), and add singleton dimensions to the result for those
    dimensions.
    """
    #This will be applied in dist.py to distribution arguments, 
    #which may be non-tensors.  These non-tensors should broadcast 
    #properly whatever happens, so we can return immediately.
    if not isinstance(x, Tensor):
        assert isinstance(x, Number)
        return x

    assert_no_ellipsis(dims)

    x_dims = set(generic_dims(x))

    dims_in_x = [dim for dim in dims if dim in x_dims]
    x = generic_order(x, dims_in_x)

    idxs = [(slice(None) if (dim in x_dims) else None) for dim in dims]
    x = generic_getitem(x, idxs)

    return x

def dim2named_tensor(x):
    """
    Converts a torchdim to a named tensor.
    """
    dims = generic_dims(x)
    names = [repr(dim) for dim in dims]
    return generic_order(x, dims).rename(*names, ...)

#### Utilities for working with dictionaries of plates

def named2dim_tensor(d, x):
    """
    Args:
        d (dict): dictionary mapping plate name to torchdim Dim.
        x (t.Tensor): named tensor.
    Returns:
        A torchdim tensor.

    Assumes that all named dimensions appear in the dict.
    """
    #can't already be a dimtensor
    assert not is_dimtensor(x)

    #if a number then just return
    if isinstance(x, Number):
        return x

    assert isinstance(x, t.Tensor)

    for name in x.names:
        if name not in d:
            raise Exception(f"No torchdim dimension for named dimension {name} in named2dim_tensor")

    torchdims = [(slice(None) if (name is None) else d[name]) for name in x.names]

    return generic_getitem(x.rename(None), torchdims)

def named2dim_tensordict(d, tensordict):
    """Maps named2dim_tensor over a dict of tensors
    Args:
        d (dict): dictionary mapping plate name to torchdim Dim.
        tensordict (dict): dictionary mapping variable name to named tensor.
    Returns:
        dictionary mapping variable name to torchdim tensor.
    """
    return {k: named2dim_tensor(d, tensor) for (k, tensor) in tensordict.items()}


def extend_plates_with_sizes(plates, size_dict):
    """Extends a plate dict using a size dict.
    Args:
        d (plate dict): dictionary mapping plate name to torchdim Dim.
        size_dict: dictionary mapping plate name to integer size.
    Returns:
        a plate dict extended with the sizes in size_dict.
    """
    new_dict = {}
    for (name, size) in size_dict.items():
        if (name not in plates):
            new_dict[name] = Dim(name, size)
        elif size != plates[name].size:
            raise Exception(f"""Mismatch in sizes for plate '{name}',
             data has size '{size}' but model indicates size '{plates[name].size}'""")
    return {**plates, **new_dict}

def extend_plates_with_named_tensor(plates, tensor):
    """Extends a plate dict using any named dimensions in `tensor`.
    Args:
        d (plate dict): dictionary mapping plate name to torchdim Dim.
        tensor: named tensor.
    Returns:
        a plate dict extended with the sizes of the named dimensions in `tensor`.
    """
    size_dict = {name: tensor.size(name) for name in tensor.names if name is not None}
    return extend_plates_with_sizes(plates, size_dict)

def extend_plates_with_named_tensors(plates, tensors):
    """Extends a plate dict using any named dimensions in `tensors`.
    Args:
        d (plate dict): dictionary mapping plate name to torchdim Dim.
        tensors: an iterable of named tensor.
    Returns:
        a plate dict extended with the sizes of the named dimensions in `tensors`.
    """
    for tensor in tensors:
        plates = extend_plates_with_named_tensor(plates, tensor)
    return plates

def platenames2platedims(plates, platenames):
    if isinstance(platenames, str):
        platenames = (platenames,)
    return [plates[pn] for pn in platenames]


#### Utilities for tensor reductions.

def chain_reduce(f, ms, T, Kprev, Kcurr):
    ms = ms.order(T)
    assert Kprev.size == Kcurr.size
    assert Kprev in set(ms.dims)
    assert Kcurr in set(ms.dims)
    Kmid = Dim('Kmid', Kprev.size)

    while ms.shape[0] != 1:
        prev = ms[::2]
        curr = ms[1::2]
        remainder = None

        #If there's an odd number of tensors
        if len(prev) > len(curr):
            assert len(prev) == len(curr)+1
            remainder = prev[-1:]
            prev = prev[:-1]

        #Rename so that the matmul makes sense.
        prev = prev.order(Kcurr)[Kmid]
        curr = curr.order(Kprev)[Kmid]

        ms = f(prev, curr, Kmid)
        if remainder is not None:
            ms = t.cat([ms, remainder], 0)

    return ms[0]

def td_matmul(prev, curr, Kmid):
    return (prev*curr).sum(Kmid)

def logmmexp(prev, curr, Kmid):
    max_prev = max_dims(prev, (Kmid,))
    max_curr = max_dims(curr, (Kmid,))

    exp_prev_minus_max = (prev - max_prev).exp()
    exp_curr_minus_max = (curr - max_curr).exp()

    result_minus_max = td_matmul(exp_prev_minus_max, exp_curr_minus_max, Kmid).log()
    return result_minus_max + max_prev + max_curr

def chain_logmmexp(ms, T, Kprev, Kcurr):
    return chain_reduce(logmmexp, ms, T, Kprev, Kcurr)


def reduce_Ks(tensors, Ks_to_sum):
    """
    Fundamental method that sums over Ks
    """
    maxes = [max_dims(tensor, Ks_to_sum) for tensor in tensors]
    # add a tiny amount for numerical stability
    tensors_minus_max = [(tensor - m).exp() + 1e-15 for (tensor, m) in zip(tensors, maxes)]
    result = torchdim_einsum(tensors_minus_max, Ks_to_sum).log()

    if 0<len(Ks_to_sum):
        result = result - sum(math.log(K.size) for K in Ks_to_sum) #t.log(t.tensor([K.size for K in Ks_to_sum])).sum()#.to(device=result.device)
    return sum([result, *maxes])

def max_dims(x, dims):
    #Ignore dims that aren't in the tensors
    set_xdims = set(generic_dims(x))
    dims = [dim for dim in dims if dim in set_xdims]
    if 0!=len(dims):
        x = x.order(dims).max(0).values
    return x
#
#    if 0==len(dims):
#        return x
#    else:
#        return generic_order(x, dims).flatten(0, len(dims)-1).max(0).values

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
        dims = generic_dims(tensor)
        arg_idxs.append([dim_to_idx[dim] for dim in dims])
        undim_tensors.append(generic_order(tensor, dims))

    assert all(not is_dimtensor(tensor) for tensor in undim_tensors)

    einsum_args = [val for pair in zip(undim_tensors, arg_idxs) for val in pair] + [out_idxs]

    result = t.einsum(*einsum_args)
    if 0 < len(out_dims):
        result = result[out_dims]

    return result
