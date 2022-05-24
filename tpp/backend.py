#Write Gibbs sampler!!

import torch as t
import opt_einsum as oe
import torch.distributions as td
import tpp

K_prefix = "K_"
plate_prefix = "plate_"

def is_K(dim):
    """
    Check that dim is correctly marked as 'K_'
    """
    if dim is None:
        return False
    return dim[:len(K_prefix)] == K_prefix

def is_plate(dim):
    """
    Check that dim is correctly marked as 'plate_'
    """
    if dim is None:
        return False
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
    """
    Partitions a list of tensors into two sets, one containing a given dim_name,
    one that doesn't
    """
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
    """
    Returns unique ordered list of names for tensors in args
    """
    return ordered_unique([name for arg in args for name in arg.names])

def align_tensors(args):
    """
    Aligns tensors in args
    """
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

    result_lp = (result_p.log() + sum(max_lps)).squeeze(K_name)

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

def sum_lps(lps):
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
    assert 1==len(lps)
    assert 1==lps[0].numel()
    return lps[0], marginals

def sum_none_dims(lp):
    """
    Sum over None dims in lp
    """
    none_dims = [i for i in range(len(lp.names)) if lp.names[i] is None]
    if 0 != len(none_dims):
        lp = lp.sum(none_dims)
    return lp

def sum_logpqs(logps, logqs):
    """
    Arguments:
        logps: dict{rv_name -> log-probability tensor}
        logqs: dict{rv_name -> log-probability tensor}
    Returns:
        elbo, used for VI
        marginals: [(K_dim, list of marginal log-probability tensors)], used for Gibbs sampling
    """

    assert len(logqs) <= len(logps)

    # check all named dimensions in logps are either positional, plates or "K"
    for lp in logqs.values():
        for n in lp.names:
            assert (n is None) or n=="K" or is_plate(n)

    # convert K
    logqs = {n:lp.rename(K=K_prefix+n) for (n, lp) in logqs.items()}

    # check all named dimensions in logps are either positional, plates or Ks
    for lp in logps.values():
        for n in lp.names:
            assert (n is None) or is_K(n) or is_plate(n)

    # sum over all non-plate and non-K dimensions
    logps = {rv: sum_none_dims(lp) for (rv, lp) in logps.items()}
    logqs = {rv: sum_none_dims(lp) for (rv, lp) in logqs.items()}

    # sanity checking for latents (only latents appear in logqs)
    for rv in logqs:
        #check that any rv in logqs is also in logps
        assert rv in logps

        lp = logps[rv]
        lq = logqs[rv]

        # check same plates appear in lp and lq
        lp_plates = [n for n in lp.names if is_plate(n)]
        lq_plates = [n for n in lq.names if is_plate(n)]
        assert set(lp_plates) == set(lq_plates)

        # check there is a K_name corresponding to rv name in both tensors
        assert K_prefix+rv in lp.names
        assert K_prefix+rv in lq.names


    #combine all lps, negating logqs
    all_lps = list(logps.values()) + [-lq for lq in logqs.values()]

    return sum_lps(all_lps)

def vi(logps, logqs):
    elbo, _ = sum_logpqs(logps, logqs)
    return elbo


def gibbs(marginals):
    #names of random variables that we've sampled
    K_names = []
    #corresponding sampled indexes
    ks = []
    for (rv, log_margs) in marginals[::-1]:
        #throw away log_margs without dimension of interest
        K_name = rv

        log_margs = [lm for lm in log_margs if (K_name in lm.names)]
        #index into log_margs with previously sampled ks
        #different indexing behaviour for tuples vs lists
        log_margs = [lm.align_to(*K_names, '...')[tuple(ks)] for lm in log_margs]

        #the only K left should be K_name
        #and plates should all be the same (and there should be more than one tensor)
        #therefore all dim_names should be the same,
        dmss = [set(lm.names) for lm in log_margs]
        dms0 = dmss[0]
        for dms in dmss[1:]:
            assert dms0 == dms

        #the only K left should be K_name
        remaining_K_names = [n for n in dms0 if is_K(n)]
        assert 1==len(remaining_K_names)
        assert K_name == remaining_K_names[0]

        #align and combine tensors
        plate_names = [n for n in dms0 if is_plate(n)]
        align_names = plate_names + remaining_K_names
        lp = sum([lm.align_to(*align_names) for lm in log_margs])

        #add K_name and sample to lists
        K_names.append(remaining_K_names[0])
        ks.append(td.Categorical(lp.rename(None)).sample())

    #return a dictionary of random variable names
    return {K_name[len(K_prefix):] : k for (K_name, k) in zip(K_names, ks)}






if __name__ == "__main__":
    a = t.randn(3,3).refine_names('K_d', 'K_b')
    ap = t.randn(3,3).refine_names('K_b', 'K_a')
    b = t.randn(3,3,3).refine_names('K_a', 'K_b', 'plate_s')
    c = t.randn(3,3,3).refine_names('K_c', 'K_d', 'plate_s')
    d = t.randn(3,3,3).refine_names('K_a', 'K_c', 'plate_b')
    lps = (a,b,c,d)

    assert t.allclose((a.exp() @ ap.exp()/3).log().rename(None), reduce_K([a, ap], 'K_b')[0].rename(None))

    lp, marginals = sum_lps(lps)
    print(gibbs(marginals))
