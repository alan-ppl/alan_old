import torch as t
import opt_einsum as oe

def args_with_dim_name(args, dim_name):
    return [arg for arg in args if isinstance(arg, t.Tensor) and dim_name in args.names]

def unify_names(args):
    return list({name for arg in args for name in arg.names})

def align_tensors(args):
    unified_names = unify_names(args)
    return [arg.align_to(*unified_names) for arg in args]

def reduce_dim(all_lps, dim_name):
    """
    Takes a the full list of tensors and a dim_name to reduce over.
    Splits the list into those with and without that dimension.
    Combines all tensors with that dim.
    Returns a list of the tensors without dim_name + the combined tensor.
    """
    #partition into those with dim_name and those without it
    lps       = [lp for lp in all_lps if dim_name     in lp.names]
    other_lps = [lp for lp in all_lps if dim_name not in lp.names]
     
    lps = align_tensors(lps)
    K = lps[0].size(dim_name)

    max_lps = [lp.max(dim_name, keepdim=True)[0] for lp in lps]
    norm_ps = [(lp - mlp).exp() for (lp, mlp) in zip(lps, max_lps)]

    result_p = norm_ps[0]
    for norm_p in norm_ps[1:]:
        result_p = result_p * norm_p
    result_p = result_p.sum(dim_name, keepdim=True) / K

    result_lp = (result_p.log() - sum(max_lps)).squeeze(dim_name)
    other_lps.append(result_lp)
    return other_lps

def reduce_dims(lps, dim_names):
    """
    Takes a list of log-probability tensors, and a list of dimension names to do the reductions.

    Returns the full sum (for the variational objective) and a list of lists of
    partially reduced tensors representing marginal probabilities for Gibbs sampling.
    """
    result = []
    for dim_name in dim_names:
        result.append((dim_name, lps))
        lps = reduce_dim(lps, dim_name)
    return lps, result

def oe_dim_order(lps):
    """
    reduce_dims, where the ordering is chosen using einsum.
    first, convert dim_names to opt_einsum symbols
    """
    unified_names = unify_names(lps)
    name2sym = dict() 
    sym2name = dict() 
    for (i, name) in enumerate(unified_names):
        sym = oe.get_symbol(i)
        name2sym[name] = sym
        sym2name[sym] = name

    lp_syms = []
    for lp in lps:
        list_syms = [name2sym[n] for n in lp.names]
        lp_syms.append(''.join(list_syms))

    #Keep plate names, i.e. those without "K_" at start
    out_names = [n for n in unified_names if n[:2] != "K_"]
    list_out_syms = [name2sym[n] for n in out_names]
    out_syms = ''.join(list_out_syms) 
    subscripts = ','.join(lp_syms) + '->' + out_syms

    ce = oe.contract_expression(subscripts, *[lp.shape for lp in lps])
    dim_names = []
    for (_, syms, _, _, _) in ce.contraction_list:
        for sym in syms:
            dim_names.append(sym2name[sym])
    return dim_names
     

if __name__ == "__main__":
    a = t.randn(3,3,3).refine_names('K_a', 'K_b', 'P_s')
    b = t.randn(3,3,3).refine_names('K_c', 'K_d', 'P_s')
    c = t.randn(3,3,3).refine_names('K_a', 'K_c', 'P_b')
    d = t.randn(3,3).refine_names('K_d', 'K_b')
    lps = (a,b,c,d)
    order = oe_dim_order(lps)

    lp, lists = reduce_dims(lps, order)

