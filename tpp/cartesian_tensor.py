import torch
import torch.nn.functional as F

from functorch import vmap


torch._C._debug_only_display_vmap_fallback_warnings(True)
override_dict = torch.overrides.get_testing_overrides()


# Copied from `Reduction Ops` section in https://pytorch.org/docs/stable/torch.html/
vmap_ops = [
    "dot",
    "outer",
    "linear",
    "conv2d",
    "argmax",
    "argmin",
    "amax",
    "amin",
    "all",
    "any",
    "max",
    "min",
    "dist",
    "logsumexp",
    "mean",
    "median",
    "nanmedian",
    "mode",
    "norm",
    "nansum",
    "prod",
    "quantile",
    "nanquantile",
    "std",
    "std_mean",
    "sum",
    "unique",
    "unique_consecutive",
    "var",
    "var_mean",
    "count_nonzero"
]


def expand(arg, shape):
    """
    Usual expand, but allows len(shape) != len(arg.shape), by expanding only the leading dimensions
    """
    return arg.expand(*shape, *arg.shape[-(len(arg.shape) - len(shape)):])


def typemap(f, typ, args, kwargs):
    """
    Apply a function to all args and kwargs of a given type
    """
    args = [(f(arg) if isinstance(arg, typ) else arg) for arg in args]
    kwargs = {key: (f(val) if isinstance(val, typ) else val) for (key, val) in kwargs.items()}
    return args, kwargs


def cartesiantensormap(f, args, kwargs):
    return typemap(f, CartesianTensor, args, kwargs)


def tensormap(f, args, kwargs):
    return typemap(f, torch.Tensor, args, kwargs)


def tensors(args, kwargs):
    """
    Extract list of all tensors from args, kwargs
    """
    return [arg for arg in [*args, *kwargs.values()] if isinstance(arg, torch.Tensor)]


def cartesian_tensorfy_value(val, unified_names):
    """
    Sometimes, pytorch operations return (nested) tuples of tensors.
    Go through tree, and convert all Tensors to CartesianTensors
    """
    if isinstance(val, torch.Tensor):
        return CartesianTensor(val.refine_names(*unified_names, ...))
    elif isinstance(val, tuple):
        return tuple(cartesian_tensorfy_value(v, unified_names) for v in val)
    else:
        return val


def pad_nones(arg, max_pos_dim):
    """
    Pad with as many positional dimensions as necessary after named dimensions
    to reach max_pos_dim
    """
    names = arg.names
    # current number of None's
    pos_dim = sum(name is None for name in names)
    # current number of named dimensions (excluding None's)
    named_dims = len(names) - pos_dim

    # strip names because unsqueeze can't cope with them
    arg = arg.rename(None)
    for _ in range(max_pos_dim-pos_dim):
        arg = arg.unsqueeze(named_dims)

    return arg.refine_names(*names, ...)


class CartesianTensor(torch.Tensor):
    def __init__(self, tensor):
        self._t = tensor

    def __repr__(self):
        return "CartesianTensor" + self._t.__repr__()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Convert CartesianTensor to Tensor to avoid recursive wrapper call.
        args, kwargs = cartesiantensormap(lambda x: x._t, args, kwargs)

        # Sorted list of all unique names
        unified_names = set([name for arg in tensors(args, kwargs) for name in arg.names])
        unified_names.discard(None)
        unified_names = sorted(unified_names)

        # Align tensors onto that sorted list
        args, kwargs = tensormap(lambda x: x.align_to(*unified_names, ...), args, kwargs)

        if func.__name__ in vmap_ops:
            # Expand all named dimensions to be the same size for vmap
            max_shape = [
                max(arg.size(name) for arg in tensors(args, kwargs)) for name in unified_names
            ]
            args, kwargs = tensormap(lambda x: expand(x, max_shape), args, kwargs)
            # Strip names from arguments because vmap can't cope with them
            args, kwargs = tensormap(lambda x: x.rename(*(len(x.shape)*[None])), args, kwargs)

            # vmap wants in_dim=0 for tensors and in_dim=None for non-tensors
            in_dims = tuple([(0 if isinstance(arg, torch.Tensor) else None) for arg in args])
            # Apply vmap once for every named argument
            for i in range(len(unified_names)):
                func = vmap(func, in_dims=in_dims)

        else:
            # pad with number of none's required to make broadcasing work
            max_pos_dim = max(
                sum(name is None for name in arg.names) for arg in tensors(args, kwargs)
            )
            args, kwargs = tensormap(lambda x: pad_nones(x, max_pos_dim), args, kwargs)

        val = func(*args, **kwargs)
        return cartesian_tensorfy_value(val, unified_names)
