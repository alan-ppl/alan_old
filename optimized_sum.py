import torch as t
import opt_einsum as oe

K_prefix = "K_"
plate_prefix = "plate_"

def is_K(dim):
    return dim[:len(K_prefix)] == K_prefix
def is_plate(dim):
    return dim[:len(plate_prefix)] == plate_prefix

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

def partition_tensors(lps, dim_name):
    has_dim = [lp for lp in lps if dim_name     in lp.names]
    no_dim  = [lp for lp in lps if dim_name not in lp.names]
    return has_dim, no_dim

def args_with_dim_name(args, dim_name):
    """
    Arguments:
        args: list of named tensors
        dim_name: string
    Pulls out all the tensors with a dimension name == dim_name
    Returns:
        args: list of named tensors
    """
    return [arg for arg in args if isinstance(arg, t.Tensor) and dim_name in args.names]

def unify_names(args):
    return ordered_unique([name for arg in args for name in arg.names])

def align_tensors(args):
    unified_names = unify_names(args)
    return [arg.align_to(*unified_names) for arg in args]

def reduce_K(all_lps, K_name):
    """
    Takes a the full list of tensors and a K_name to reduce over.
    Splits the list into those with and without that dimension.
    Combines all tensors with that dim.
    Returns a list of the tensors without dim_name + the combined tensor.
    """
    lps_with_K, other_lps = partition_tensors(all_lps, K_name)

    lps_with_K = align_tensors(lps_with_K)

    max_lps = [lp.max(K_name, keepdim=True)[0] for lp in lps_with_K]
    norm_ps = [(lp - mlp).exp() for (lp, mlp) in zip(lps_with_K, max_lps)]

    result_p = norm_ps[0]
    for norm_p in norm_ps[1:]:
        result_p = result_p * norm_p
    result_p = result_p.mean(K_name, keepdim=True)

    result_lp = (result_p.log() - sum(max_lps)).squeeze(K_name)

    other_lps.append(result_lp)
    return other_lps

def reduce_Ks(lps, Ks_to_keep):
    """
    Takes a list of log-probability tensors, and a list of dimension names to do the reductions.

    Returns the full sum (for the variational objective) and a list of lists of
    partially reduced tensors representing marginal probabilities for Gibbs sampling.
    Arguments:
        lps: List of tensors within a plate
        K_dims_to_keep: List of K_dims to keep because they appear in higher-level plates.
    Returns:
        lps: log-probability tensors with all K's that just appear in this plate summed out
        List of (K_dim, tensors) pairs
    """
    marginals = []
    ordered_Ks = oe_dim_order(lps, Ks_to_keep)

    for K_name in ordered_Ks:
        marginals.append((K_name, lps))
        lps = reduce_K(lps, K_name)
    return lps, marginals

def oe_dim_order(lps, Ks_to_keep):
    """
    Returns an optimized dimension ordering for reduction.
    Arguments:
        lps: List of tensors within a plate
        K_dims_to_keep: List of K_dims to keep because they appear in higher-level plates.
    Returns:
        dim_order
    """
    #create backward and forward mappings from all tensor names to opt_einsum symbols
    unified_names = unify_names(lps)
    name2sym = dict() 
    sym2name = dict() 
    for (i, name) in enumerate(unified_names):
        sym = oe.get_symbol(i)
        name2sym[name] = sym
        sym2name[sym] = name

    #create opt_einsum subscripts formula for inputs
    lp_syms = []
    for lp in lps:
        list_syms = [name2sym[n] for n in lp.names]
        lp_syms.append(''.join(list_syms))

    #create opt_einsum subscripts formula for outputs
    #don't sum over plates or K_dims_to_keep, so
    #so they must appear in the output subscripts
    out_names = [n for n in unified_names if (not is_K(n) or (n in Ks_to_keep))]
    list_out_syms = [name2sym[n] for n in out_names]
    out_syms = ''.join(list_out_syms) 
    subscripts = ','.join(lp_syms) + '->' + out_syms

    #extract order from opt_einsum, and convert back to names
    ce = oe.contract_expression(subscripts, *[lp.shape for lp in lps])
    ordered_Ks = []
    for (_, syms, _, _, _) in ce.contraction_list:
        for sym in syms:
            ordered_Ks.append(sym2name[sym])
    return ordered_Ks

     
def sum_plate(all_lps, plate_name=None):
    """
    Arguments:
        lps: full list of log-probability tensors
        plate_name
    Returns:
        lps: full list of log-probability tensors with plate_name summed out
        list: possibly empty list for summing over each dimension
    """
    if plate_name is not None:
        #partition tensors into those with/without plate_name
        lower_lps, higher_lps = partition_tensors(all_lps, plate_name)
    else:
        #top-level (no plates)
        lower_lps = all_lps
        higher_lps = []

    #collect K's that appear in higher plates
    Ks_to_keep = [n for n in unify_names(higher_lps) if is_K(n)]
 
    #sum over the K's that don't appear in higher plates
    lower_lps, marginals = reduce_Ks(lower_lps, Ks_to_keep)

    if plate_name is not None:
        #if we aren't at the top-level, sum over the plate to eliminate plate_name
        lower_lps = [l.sum(plate_name) for l in lower_lps]

    #append to all the higher-plate tensors
    higher_lps = higher_lps + lower_lps

    return higher_lps, marginals

def sum_plates(lps):
    """
    The final, exported function.
    Arguments:
        lps: full list of log-probability tensors
    Returns:
        elbo, used for VI
        marginals: [(K_dim, list of marginal log-probability tensors)], used for Gibbs sampling
    """
    #ordered list of plates
    plate_names = [n for n in unify_names(lps) if is_plate(n)]
    #add top-layer (no plate)
    plate_names = [None] + plate_names

    #iterate from lowest plate
    marginals = []
    for plate_name in plate_names[::-1]:
        lps, _m = sum_plate(lps, plate_name)
        marginals = marginals + _m
    print(lps)
    assert 1==len(lps) 
    assert 1==lps[0].numel()
    return lps[0], marginals
        
    

if __name__ == "__main__":
    a = t.randn(3,3).refine_names('K_d', 'K_b')
    b = t.randn(3,3,3).refine_names('K_a', 'K_b', 'plate_s')
    c = t.randn(3,3,3).refine_names('K_c', 'K_d', 'plate_s')
    d = t.randn(3,3,3).refine_names('K_a', 'K_c', 'plate_b')
    lps = (a,b,c,d)

    lp, marginals = sum_plates(lps)
