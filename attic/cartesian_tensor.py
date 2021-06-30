import torch
from torch import vmap
import itertools
import inspect
from functools import partial

torch._C._debug_only_display_vmap_fallback_warnings(True)
override_dict = torch.overrides.get_testing_overrides()

# Util functions


def _is_binary_op(func):
    if func not in override_dict:
        return False
    if func == torch.einsum:
        raise NotImplementedError()
    if func == torch.outer:
        # Lots of special cases to handle... need a more elegant solution
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


def _process_common_names(lhs, rhs):
    """
    Find duplicate elements in `lhs` and `rhs` and move them to the end of two lists.
    Return three dictionaries:
    unique elements in `lhs` and `rhs`, common elements in `lhs` and `rhs`
    """
    # Find identical (name, shape) pairs in `lhs` and `rhs`.
    common_elements_set = lhs.items() & rhs.items()
    lhs_unique_set = lhs.items() - common_elements_set
    rhs_unique_set = rhs.items() - common_elements_set
    return (
        dict(lhs_unique_set),
        dict(rhs_unique_set),
        dict(common_elements_set)
    )


def _name_shape_encoder(name_shape_dict, prefix='____'):
    '''
    Convert a name_shape dictionaries to a string
    '''
    assert prefix[0] == '_'
    assert prefix[3] == '_'
    return prefix + '__'.join(
        map(lambda t: f'{t[0]}_{t[1]}', name_shape_dict.items())
    )


def _name_shape_decoder(name_shape_str):
    # name_shape_str should start with two underscores
    assert name_shape_str[0] == '_'
    assert name_shape_str[3] == '_'

    def _pair_decomposer(s):
        name, shape = s.split('_')
        return (name, int(shape))
    return list(
        map(_pair_decomposer, name_shape_str[4:].split('__'))
    )


def _get_vmap_func(func, flag):
    lhs_has_unique, rhs_has_unique, has_common = flag
    base_func = func
    if has_common:
        base_func = vmap(base_func, (0, 0), 0)
    if rhs_has_unique:
        base_func = vmap(base_func, (None, 0), 0)
    if lhs_has_unique:
        base_func = vmap(base_func, (0, None), 0)
    return base_func


def _sort_names(arg):
    plates = []
    ks = []
    for name in arg.names:
        if name is None:
            break
        if 'plate' in name:
            plates.append(name)
        elif 'K' in name:
            ks.append(name)
    return arg.align_to(*sorted(plates), *sorted(ks), ...)


class CartesianTensor(torch.Tensor):
    @classmethod
    def __preprocess_binary_inputs__(cls, lhs, rhs):
        '''
        Reorganize names. (vmap does not support named tensor)
        '''
        # Raw name_shape dict
        lhs_pk = _get_name(lhs)
        rhs_pk = _get_name(rhs)
        # Process name_shape dict
        lhs_unique, rhs_unique, common = _process_common_names(lhs_pk, rhs_pk)
        # Reorder lhs input
        lhs_reorg = lhs.align_to(*lhs_unique.keys(), *common.keys(), ...)
        if lhs_unique:
            lhs_reorg = lhs_reorg.flatten(
                tuple(lhs_unique.keys()), _name_shape_encoder(lhs_unique))
        if common:
            lhs_reorg = lhs_reorg.flatten(
                tuple(common.keys()), _name_shape_encoder(common))
        # Reorder rhs input
        rhs_reorg = rhs.align_to(*rhs_unique.keys(), *common.keys(), ...)
        if rhs_unique:
            rhs_reorg = rhs_reorg.flatten(
                tuple(rhs_unique.keys()), _name_shape_encoder(rhs_unique))
        if common:
            rhs_reorg = rhs_reorg.flatten(
                tuple(common.keys()), _name_shape_encoder(common))

        flag = (bool(lhs_unique), bool(rhs_unique), bool(common))
        return lhs_reorg, rhs_reorg, flag

    @classmethod
    def __postprocess_binary_output__(cls, out, lhs, rhs, flag):
        lhs_has_unique, rhs_has_unique, has_common = flag
        output_names = []
        # Names are erased during the computation but we could retrieve them
        # from the inputs `lhs` and `rhs`.
        if lhs_has_unique:
            output_names.append(lhs.names[0])
        if rhs_has_unique:
            output_names.append(rhs.names[0])
        if has_common:
            # Extract common names from `lhs`. (`rhs` will also do)
            if lhs_has_unique:
                # lhs: (lhs_unique, common, ...)
                output_names.append(lhs.names[1])
            else:
                # lhs: (common, ...)
                output_names.append(lhs.names[0])
        out = out.refine_names(*output_names, ...)
        for name in output_names:
            out = out.unflatten(name, _name_shape_decoder(name))
        return _sort_names(out)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Binary Ops
        if _is_binary_op(func):
            # Reorder lhs and rhs as (unique_names, common_names, user_dims)
            lhs, rhs, flag = cls.__preprocess_binary_inputs__(args[0], args[1])
            args = (lhs.rename(None), rhs.rename(None), *args[2:])
            func = _get_vmap_func(func, flag)
            # out: (lhs_unique, rhs_unique, common, ...)
            out = super().__torch_function__(func, types, args, kwargs)
            return cls.__postprocess_binary_output__(out, lhs, rhs, flag)

        return super().__torch_function__(func, types, args, kwargs)
