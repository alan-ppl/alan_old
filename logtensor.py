"""
Defines specialised transpose, tensordot and einsum, to provide an opt_einsum backend.
"""
import torch as t
import opt_einsum

def transpose(x, perm=None):
    if perm is None:
       perm = range(len(x.shape))[::-1]
    return x.permute(*perm)

def max(x, dims):
    for dim in dims:
        x = x.max(dim=dim, keepdim=True)[0]
    return x

def squeeze(x, dims):
    assert all(0 <= dim for dim in dims)
    for dim in sorted(dims, reverse=True):
        x = x.squeeze(dim)
    return x

def tensordot(x, y, axes):
    if isinstance(axes, int):
        axes = (range(x.ndim-axes, x.ndim), range(axes))
    dimx, dimy = axes

    xmax_keepdim = max(x, dimx)
    ymax_keepdim = max(y, dimy)

    xmax = squeeze(xmax_keepdim, dimx)
    ymax = squeeze(ymax_keepdim, dimy)

    xnorm = x - xmax_keepdim
    ynorm = y - ymax_keepdim

    result_norm = t.tensordot(xnorm.exp(), ynorm.exp(), axes)
    
    return result_norm.log() + xmax.view(*xmax.shape, *(ymax.ndim*[1])) + ymax

def einsum(subscripts, *args, dtype=None):
    input_dtype = args[0].dtype
    assert all(arg.dtype == input_dtype for arg in args)

    if dtype is not None:
        args = [arg.to(dtype) for arg in args]

    max_args = [arg.max() for arg in args]
    exp_norm_args = [(arg - max_arg).exp() for (arg, max_arg) in zip(args, max_args)]
    norm_result = t.einsum(subscripts, *exp_norm_args)
    result = norm_result.log() + sum(max_args)
    return result.to(dtype=input_dtype)
    

if __name__ == "__main__":
    from opt_einsum import contract

    def test_log_contract(subscripts, *args):
        exp_args = [arg.exp() for arg in args]
        return contract(subscripts, *exp_args, backend='torch').log()

    def compare_contract(subscripts, *args):
        r1 = test_log_contract(subscripts, *args)
        r2 = contract(subscripts, *args, backend='logtensor')
        return (r1, r2)

    N = 2
    X = t.randn(3, N)
    Y = t.randn(N, 4)
    r1, r2 = compare_contract('ij,jk->ik', X, Y)
    assert t.allclose(r1, r2)

    X = t.randn(N, N)
    Y = t.randn(N, N)
    r1, r2 = compare_contract('ii,jj->ij', X, Y)
    assert t.allclose(r1, r2)

    I = t.randn(N, N, N, N)
    C = t.randn(N, N)
    r1, r2 = compare_contract('pi,pj,ijkl,rk,sl->prs', C, C, I, C, C)
    assert t.allclose(r1, r2)
