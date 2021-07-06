import torch
from torch import vmap
import itertools
import inspect
from functools import partial

torch._C._debug_only_display_vmap_fallback_warnings(True)
override_dict = torch.overrides.get_testing_overrides()


# Operators that supports multi-dimensional broadcasting
_fallback_ops = [torch.add, torch.subtract, torch.multiply,
                 torch.divide, torch.matmul, torch.pow]

_special_binary_op_list = [
    torch.outer,
]


# Util functions
def _is_binary_op(func):
    if func not in override_dict:
        return False
    if func == torch.einsum:
        raise NotImplementedError()
    if func in _special_binary_op_list:
        return True
    func_params = inspect.signature(override_dict[func]).parameters.keys()
    key_words = ['other', 'exponent']
    return any(map(func_params.__contains__, key_words))


def _get_name(arg):
    plates = []
    ks = []
    for name, shape in zip(arg.names, arg.shape):
        if name is None:
            continue
        if 'plate' in name:
            plates.append((name, shape))
        elif 'K' in name:
            ks.append((name, shape))
    return dict(plates + ks)


class CartesianTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if _is_binary_op(func):
            lhs, rhs = args[0], args[1]
            unique_name_set = (set(lhs.names) | set(rhs.names))
            unique_name_set.discard(None)
            sorted_unique_name_set = sorted(
                unique_name_set
            )
            # Preprocess inputs, let the leading dimensions of lhs and rhs have the same name
            lhs = lhs.align_to(*sorted_unique_name_set, ...)
            rhs = rhs.align_to(*sorted_unique_name_set, ...)
            output_shape = []
            for l_shape, r_shape, name in zip(lhs.shape, rhs.shape, lhs.names):
                if name is None:
                    break
                output_shape.append(max(l_shape, r_shape))
            # Let the leading dimensions of lhs and rhs have the same shape
            lhs = lhs.expand(*output_shape, *lhs.shape[len(output_shape):])
            rhs = rhs.expand(*output_shape, *rhs.shape[len(output_shape):])
            lhs = lhs.flatten(
                sorted_unique_name_set,
                'vmap_dim'
            )
            rhs = rhs.flatten(
                sorted_unique_name_set,
                'vmap_dim'
            )
            if func in _fallback_ops:
                # Broadcasting can handle it all, no need for vmap
                # However, we need to pad user dimensions first.
                pad_count_left = max(lhs.dim(), rhs.dim()) - lhs.dim()
                pad_count_right = max(lhs.dim(), rhs.dim()) - rhs.dim()
                lhs = lhs.rename(None).view(-1, *((1,) * pad_count_left),
                                            *lhs.shape[1:])
                rhs = rhs.rename(None).view(-1, *((1,) * pad_count_right),
                                            *rhs.shape[1:])
                args = (lhs, rhs, *args[2:])
                out_compact = super().__torch_function__(func, types, args, kwargs)
            else:
                # Otherwise, vmap comes to rescue.
                args = (lhs.rename(None), rhs.rename(None), *args[2:])
                # Computation via vmap
                func = vmap(func, (0, 0), 0)
                out_compact = super().__torch_function__(
                    func, types, args, kwargs)  # (vmap_dim, ...)
            out_unnamed = out_compact.view(
                *output_shape, *out_compact.shape[1:])
            out_named = out_unnamed.refine_names(*sorted_unique_name_set, ...)
            return out_named

        return super().__torch_function__(func, types, args, kwargs)
