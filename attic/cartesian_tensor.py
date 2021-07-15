import torch
import torch.nn.functional as F

from functorch import vmap
import warnings


torch._C._debug_only_display_vmap_fallback_warnings(True)
override_dict = torch.overrides.get_testing_overrides()


# Operators that supports multi-dimensional broadcasting
_fallback_ops = [torch.add, torch.subtract, torch.multiply,
                 torch.divide, torch.matmul, torch.pow]


# Copied from `Reduction Ops` section in https://pytorch.org/docs/stable/torch.html/
_reduction_ops = [
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


def _is_reduction_op(func):
    return func.__name__ in _reduction_ops


def _require_cartesian(*args, **kwargs):
    """Check whether a function requires to be "cartesianized" by looking at
    the number of `torch.Tensor` in all arguments.

    Returns:
        True or False
    """
    count = 0
    for arg in (*args, *kwargs.values()):
        if isinstance(arg, torch.Tensor):
            count += 1
    return (count >= 2)


# Tensor definition
class CartesianTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if _require_cartesian(*args, **kwargs):
            in_axes = ()
            name_shape_dict = {}
            max_udf_len = -1
            # Positional argument
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    max_udf_len = max(
                        sum(map(lambda x: x is None, arg.names)),
                        max_udf_len)
                    name_shape_dict = {**name_shape_dict,
                                       **dict(zip(arg.names, arg.shape))}
                    in_axes = in_axes + (0,)
                else:
                    in_axes = in_axes + (None,)

            # FIXME: Vmap does not support keywork arguments at this moment
            # See https://github.com/facebookresearch/functorch/issues/70
            for name in list(kwargs.keys()):
                arg = kwargs[name]
                if isinstance(arg, torch.Tensor):
                    raise NotImplementedError("Keyword arguments are not supported yet.")

            name_shape_dict.pop(None, None)
            sorted_name_shape_pair = sorted(name_shape_dict.items())
            unique_sorted_names, names_shapes = zip(*sorted_name_shape_pair)

            def _arg_transformer(arg, fallback=False):
                """Transform an argument to a form suitable for cartesian computation.

                Args:
                    arg (torch.Tensor or other): Input argument
                    fallback (bool, optional): Whether to fallback to broadcast. Defaults to False.

                Returns:
                    The same as input: Processed arg
                """
                # If the input is not a tensor, we simply return it.
                if not isinstance(arg, torch.Tensor):
                    return arg
                # Step 1. Align input to unique_sorted_names,
                # missing dimensions will be automatically padded by 1.
                arg = arg.align_to(*unique_sorted_names, ...)
                # Step 2. Expand the padded named dimensions from 1 to their real size.
                arg = arg.expand(*names_shapes, *arg.shape[len(names_shapes):])
                # Step 3. Squeeze named dimensions into one giant dimension
                # such that we only need to invoke vmap once.
                arg = arg.flatten(unique_sorted_names, 'vmap_dim').rename(None)
                # Step 4. In case we are falling back to broadcasting mechanism,
                # we need to further align users' dimension.
                # e.g. We have x: (Ka, 2, 3), y: (Ka, 3), here we would change y to (Ka, 1, 3),
                # in order to allow broadcasting
                if fallback:
                    pad_count = max_udf_len - (arg.dim() - 1)
                    arg = arg.view(-1, *((1,) * pad_count),
                                   *arg.shape[1:])
                return arg

            if func in _fallback_ops:
                # Use broadcasting
                use_fallback = True
            else:
                # Use vmap
                use_fallback = False
                func = vmap(func, in_dims=(*in_axes,), out_dims=0)

            args = *(_arg_transformer(arg, use_fallback) for arg in args),
            out_compact = super().__torch_function__(
                func, types, args, kwargs)  # (vmap_dim, ...)
            out_unnamed = out_compact.view(
                *names_shapes, *out_compact.shape[1:])
            out_named = out_unnamed.refine_names(*unique_sorted_names, ...)
            return out_named

        if _is_reduction_op(func):
            tensor_arg_counter = 0
            in_axes = ()
            name_shape_dict = {}
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    in_axes += (0,)
                    tensor_arg_counter += 1
                    if tensor_arg_counter > 1:
                        raise ValueError(
                            "Reduction Ops is expected to have only 1 tensor arg")
                    name_shape_dict = {**name_shape_dict,
                                       **dict(zip(arg.names, arg.shape))}
                    name_shape_dict.pop(None, None)
                    sorted_name_shape_pair = sorted(name_shape_dict.items())
                    unique_sorted_names, names_shapes = zip(
                        *sorted_name_shape_pair)
                else:
                    in_axes += (None,)
            
            # FIXME: Vmap does not support keywork arguments at this moment
            # See https://github.com/facebookresearch/functorch/issues/70
            for name in list(kwargs.keys()):
                arg = kwargs[name]
                if isinstance(arg, torch.Tensor):
                    raise NotImplementedError("Keyword arguments are not supported yet.")

            if tensor_arg_counter == 0:
                raise ValueError(
                    "Reduction Ops is expected to have only 1 tensor arg")

            def _reduction_arg_transformer(arg):
                if not isinstance(arg, torch.Tensor):
                    return arg
                arg = arg.flatten(unique_sorted_names, 'vmap_dim').rename(None)
                return arg

            args = *(_reduction_arg_transformer(arg) for arg in args),
            kwargs = {k: _reduction_arg_transformer(v)
                      for k, v in kwargs.items()}
            func = vmap(func, in_axes, 0)
            out_compact = super().__torch_function__(
                func, types, args, kwargs)  # (vmap_dim, ...)
            out_unnamed = out_compact.view(
                *names_shapes, *out_compact.shape[1:])
            out_named = out_unnamed.refine_names(*unique_sorted_names, ...)
            return out_named

        return super().__torch_function__(func, types, args, kwargs)
