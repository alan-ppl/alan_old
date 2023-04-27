import math 
import torch as t
from functorch.dim import Tensor, Dim

#### Utilities for working with torchdims

def sum_non_dim(x):
    """
    Sums over all non-torchdim dimensions.
    """
    return x.sum() if x.ndim > 0 else x


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
    def inner(x, dims, ignore_extra_dims=False):
        assert_unique_dim_iter(dims)

        set_x_dims = set(generic_dims(x)) 
        if ignore_extra_dims:
            dims = tuple(dim for dim in dims if dim in set_x_dims)

        if not all(dim in set_x_dims for dim in dims):
            raise Exception("dims provided that aren't in x; can ignore them by providing ignore_extra_dims=True kwarg")

        if 0<len(dims):
            x = func(x.order(dims), 0)
        return x
    return inner

def sum_dims(x, dims):
    """
    """
    dims = list(dims)
    if 0<len(dims):
        x = x.order(dims).sum(0)
    return x

def mean_dims(x, dims):
    """
    """
    dims = list(dims)
    if 0<len(dims):
        x = x.order(dims).mean(0)
    return x

# sum_dims        = reduce_dims(t.sum)
prod_dims       = reduce_dims(t.prod)
# mean_dims       = reduce_dims(t.mean)
min_dims        = reduce_dims(lambda x, dim: t.min(x, dim).values)
max_dims        = reduce_dims(lambda x, dim: t.max(x, dim).values)
logsumexp_dims  = reduce_dims(t.logsumexp)

def logmeanexp_dims(x, dims):
    return logsumexp_dims(x, dims) - sum([math.log(dim.size) for dim in dims])


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
    #return x.dims if isinstance(x, Tensor) else ()
    return x.dims if hasattr(x, "dims") else ()

def generic_order(x, dims):
    """
    Implements x.order(dims), which is only defined for torchdim tensors
    """
    #Drop trailing Ellipsis.
    if (0 < len(dims)) and (dims[-1] == Ellipsis):
        dims = dims[:-1]

    #If x is a torch tensor, then we can't have any dims.
    if isinstance(x, t.Tensor):
        assert 0 == len(dims)

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
    Converts a torchdim to a named tensor.
    Will fail if duplicated dim names passed in
    """
    if dims is None:
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
    """

    if 0==x.ndim:
        return x

    torchdims = [(slice(None) if (name is None) else d[name]) for name in x.names]

    if all(x is None for x in x.names):
        return x

    return x.rename(None)[torchdims]

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
             data has size '{plates[name].size}' but model indicates size '{size}'""")
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

if __name__ == "__main__":
    from functorch.dim import dims

    #chain_reduce and td_matmul
    T, Kprev, K = dims(3, [5, 3, 3])
    tensor_ms = t.randn(5, 3, 3)

    tensor_result = t.chain_matmul(*t.unbind(tensor_ms, 0))
    td_result = chain_reduce(td_matmul, tensor_ms[T, Kprev, K], T, Kprev, K).order(Kprev, K)
    assert t.allclose(tensor_result, td_result)

    #logmmexp
    Kmid = dims(1, [3])
    A = t.randn(3,3)
    B = t.randn(3,3)

    tensor_result = (t.exp(A) @ t.exp(B)).log()
    td_result = logmmexp(A[Kprev, Kmid], B[Kmid, K], Kmid).order(Kprev, K)
    assert t.allclose(tensor_result, td_result)


def reduce_Ks(tensors, Ks_to_sum):
    """
    Fundamental method that sums over Ks
    """
    maxes = [max_dims(tensor, Ks_to_sum) for tensor in tensors]
    # add a tiny amount for numerical stability
    tensors_minus_max = [(tensor - m).exp() + 1e-15 for (tensor, m) in zip(tensors, maxes)]
    # print(Ks_to_sum)
    # print(tensors_minus_max)
    result = torchdim_einsum(tensors_minus_max, Ks_to_sum).log()

    #if 0<len(Ks_to_sum):
    #    result = result - t.log(t.tensor([K.size for K in Ks_to_sum])).sum().to(device=result.device)
    return sum([result, *maxes])

def max_dims(x, dims):
    """
    Note that this _keeps_ dims, and maxes over everything else, which
    goes against the usual convention
    """

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
        dims = generic_dims(tensor)
        arg_idxs.append([dim_to_idx[dim] for dim in dims])
        undim_tensors.append(generic_order(tensor, dims))

    assert all(not is_dimtensor(tensor) for tensor in undim_tensors)

    einsum_args = [val for pair in zip(undim_tensors, arg_idxs) for val in pair] + [out_idxs]
    # print(einsum_args)
    result = t.einsum(*einsum_args)
    if 0 < len(out_dims):
        result = result[out_dims]

    return result
