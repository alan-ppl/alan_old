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
    Partitions a list of tensors into two sets, one containing a given dim_name
    or only has interactions with tensors that have that dim name,
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


def reduce_Ks(lps, Ks_to_keep):
    """
    Takes a list of log-probability tensors, and a list of dimension names to do the reductions.

    Returns the full sum (for the variational objective) and a list of lists of
    partially reduced tensors representing marginal probabilities for Gibbs sampling.
    Arguments:
        lps: List of tensors within a plate
        Ks_to_keep: List of K_dims to keep because they appear in higher-level plates.
    Returns:
        output: a single log-probability tensor with all K's appearing only in this plate summed out
    """
    all_names = unify_names(lps) 
    #Make sure Ks_to_keep is restricted to Ks which are actually on lps
    Ks_to_keep = list(set(Ks_to_keep).intersection(all_names))
    #keep everything that isn't a K, and everything that we explicitly ask to keep
    names_to_keep = [name for name in all_names if not is_K(name)] + Ks_to_keep

    #Pytorch's weird einsum syntax requires us to assign an integer to each dimension
    name_to_idx = {name : idx for (idx, name) in enumerate(all_names)}
    lps_idxs = [[name_to_idx[name] for name in lp.names] for lp in lps]
    out_idxs = [name_to_idx[name] for name in names_to_keep]

    #subtract max to ensure numerical stability
    maxes = [max_dims(lp, set(lp.names).difference(names_to_keep)) for lp in lps]
    ps_minus_max = [(lp - max_.align_as(lp)).exp().rename(None) for (lp, max_) in zip(lps, maxes)]

    #Interleaves lp and list of indices for the weird PyTorch einsum syntax.
    args = [val for pair in zip(ps_minus_max, lps_idxs) for val in pair] + [out_idxs]

    output = t.einsum(*args).rename(*names_to_keep).log() 

    num_reductions = len(all_names) - len(output.names)
    if 0 < num_reductions:
        for lp in lps:
            if any(is_K(name) for name in lp.names):
                K_name = next(name for name in lp.names if is_K(name))
                K = lp.size(K_name)
                break
        output = output - num_reductions*t.log(t.tensor(K))

    for max_ in maxes:
        output = output + max_.align_as(output)

    return [output]

def max_dims(x, dims):
    if 0 == len(dims):
        return x
    elif 1 == len(dims):
        #this branch shouldn't be necessary, but there's a bug in flatten for named tensors with one dim
        max_ = self.x.max(dims[0]).values
    else:
        ordered_dims = tuple(name for name in x.names if name in dims)
        return x.flatten(ordered_dims, 'flattened').max('flattened').values

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

    lower_lps = reduce_Ks(lower_lps, Ks_to_keep)

    if plate_name is not None:
        #if we aren't at the top-level, sum over the plate to eliminate plate_name
        lower_lps = [l.sum(plate_name) if plate_name in l.names else l for l in lower_lps]



    #append to all the higher-plate tensors
    higher_lps = higher_lps + lower_lps


    return higher_lps

def sum_tensors(lps):
    """
    The final, exported function.
    Arguments:
        lps: full list of log-probability tensors
    Returns:
        elbo, used for VI
        marginals: [(K_dim, list of marginal log-probability tensors)], used for Gibbs sampling
    """
    #ordered list of plates
    # NOTE: This assumes the lps come in the right order! There should be an assert here.
    plate_names = [n for n in unify_names(lps) if is_plate(n)]
    #add top-layer (no plate)
    plate_names = [None] + plate_names

    #iterate from lowest plate
    for plate_name in plate_names[::-1]:
        lps = sum_plate(lps, plate_name)
        # print(plate_name)
        # print(lps)

    assert 1==len(lps)
    #assert 0==len(sum(lps).shape)
    lp = lps[0]
    assert 1==lp.numel()

    return lp#sum(lps)



