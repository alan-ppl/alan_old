import torch as t
import numpy as np
import random
from numpy.lib.stride_tricks import as_strided

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)

def n_mean(arr, mean_no):
    #arr = np.asarray(lst)
    arr_reduced = block_reduce(arr, block_size=(1,mean_no), func=np.mean, cval=np.mean(arr))
    return arr_reduced

def block_reduce(arr, block_size=2, func=np.sum, cval=0, func_kwargs=None):
    """
    from scikit-image
    """

    if np.isscalar(block_size):
        block_size = (block_size,) * arr.ndim
    elif len(block_size) != arr.ndim:
        raise ValueError("`block_size` must be a scalar or have "
                         "the same length as `arr.shape`")

    if func_kwargs is None:
        func_kwargs = {}

    pad_width = []
    for i in range(len(block_size)):
        if block_size[i] < 1:
            raise ValueError("Down-sampling factors must be >= 1. Use "
                             "`skarr.transform.resize` to up-sample an "
                             "arr.")
        if arr.shape[i] % block_size[i] != 0:
            after_width = block_size[i] - (arr.shape[i] % block_size[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))

    arr = np.pad(arr, pad_width=pad_width, mode='constant',
                   constant_values=cval)

    blocked = view_as_blocks(arr, block_size)

    return func(blocked, axis=tuple(range(arr.ndim, blocked.ndim)),
                **func_kwargs)

def view_as_blocks(arr_in, block_shape):
    """
    from scikit-image
    """
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')

    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")

    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length "
                         "as 'arr_in.shape'")

    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")

    # -- restride the array to build the block view
    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out
