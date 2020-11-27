import random
import torch as t
import tpp_trace as tra
import utils as u
import math


# TODO: add data handling
# TODO: add plates
def combine_tensors(tensors_dict) :    
    """
        :param tensors_dict: dict of log_prob tensors 
        :return: scalar of log_prob tensors, multiplied and summed out dims
    """
    # 1. Take all indices I in the tensors 
    # make a unified set of them
    all_names = get_all_names(tensors_dict)
    k_dims = filter_names(all_names, plates=False)
    
    # 2. if anything is left in `pos_` dims, sum out
    # TODO: remove; doing this in forward now, before `combine_tensors`.
    tensors_dict = clear_user_dims(tensors_dict, all_names)
    
    # 3. for each k dim, get tensors that depend on dim and squash them
    while k_dims :
        random_index = random.randrange(len(k_dims))
        dim = k_dims.pop(random_index)
        tensors_dict = reduce_by_dim(tensors_dict, dim)
    
    return tensors_dict


def get_all_names(d) :
    dims = list(d.keys()) 
    all_names = [d[dim].names for dim in dims]
    
    return set([item for t in all_names \
                     for item in t])


def filter_names(names, plates=False) :
    if not plates :
        names = [name for name in names \
                  if tra.k_dim_name("") in name]
        
    return names


def clear_user_dims(d, names) :
    user_dims = [ name for name in names \
                    if tra.pos_name("") in name ]
    
    for user_dim in user_dims :
        for k, T in d.items() :
            d[k] = T.sum(user_dim)
    
    return d


def reduce_by_dim(d, dim):
    assert( tra.k_dim_name("") in dim )
    
    other_tensors, i_tensors = get_dependent_factors(d, dim)
    
    all_names = get_all_names(i_tensors)
    k_dims = filter_names(all_names, plates=False)
    
    # 4. use torch names to get all dims in same order
    for k, tensor in i_tensors.items() :
        # TODO: watch that ellipsis
        i_tensors[k] = tensor.align_to(*k_dims)#, ...)
    
    # TODO: Can just pop the first i_tensor?
    T = 0

    # 5. multiply (as sum logs)
    for k, tensor in i_tensors.items() :
        T = T + tensor
    
    nk = T.size(dim)
    # 6. sum out dim
    T = t.logsumexp(T, dim) - math.log(nk)
    T = t.tensor(T, requires_grad=True)
    # 7. put it back
    other_tensors[dim] = T
    
    return other_tensors


# old attempt at stability
def alt_reduce_by_dim(d, dim):
    assert( tra.k_dim_name("") in dim )
    
    other_tensors, i_tensors = get_dependent_factors(d, dim)
    all_names = get_all_names(i_tensors)
    k_dims = filter_names(all_names, plates=False)
    
    if i_tensors :
        # 4. use torch names to get all their dims in same order
        for k, tensor in i_tensors.items() :
            i_tensors[k] = tensor.align_to(*k_dims)#, ...)
        
        key = list(i_tensors.keys())[0]
        T = i_tensors.pop(key) 
    
        # 5. multiply 
        for k, tensor in i_tensors.items() :
            T = u.logmulmeanexp(tensor, T, dim)

        # 6. sum out dim
        T = T.sum(dim)

        other_tensors[dim] = T
    
    return other_tensors


# factors T∣K_i that depend on K_i
def get_dependent_factors(d, i):
    # TODO: consider lists instead
    dependents = { k: tensor for k, tensor in d.items() \
                    if i in tensor.names }
    nondependents = { k: tensor for k, tensor in d.items() \
                        if i not in tensor.names }
    
    return nondependents, dependents




if __name__ == "__main__" :
    kappa = 2
    n = 2
    data = {} # {"a": 4}
    tr = tra.sample_and_eval(chain_dist, draws=kappa, nProtected=n, data=data)
    tensors = tr.trace.out_dicts['log_prob']

    print(tpp.combine_tensors(tensors))