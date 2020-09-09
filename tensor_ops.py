import random
import torch as t
import tpp_trace as tra
import utils as u


# TODO: add data handling
# TODO: add plates
def combine_tensors(tensors_dict) :    
    """
        :param tensors_dict: dict of log_prob tensors 
        :return: scalar of log_prob tensors, multiplied and summed out dims
    """
    # 1. Take all indices $$I$$ in the tensors 
    # make a unified set of them
    all_names = get_all_names(tensors_dict)
    k_dims = filter_names(all_names, plates=False)
    
    # 2. if anything is left in `pos_` dims, get rid of them (sum out)
    # TODO: maybe do this before `combine_tensors`.
    tensors_dict = clear_user_dims(tensors_dict, all_names)
    
    # 3. for each k dim, get tensors that depend on dim and squash them
    while k_dims :
        random_index = random.randrange(len(k_dims))
        dim = k_dims.pop(random_index)
        tensors_dict = naive_reduce_by_dim(tensors_dict, dim)
    
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


# Probably unstable
def naive_reduce_by_dim(d, dim):
    assert( tra.k_dim_name("") in dim )
    
    other_tensors, i_tensors = get_dependent_factors(d, dim)
    
    all_names = get_all_names(i_tensors)
    k_dims = filter_names(all_names, plates=False)
    
    # 4. use torch names to get all their dims in same order
    for k, tensor in i_tensors.items() :
        # TODO: watch out for that ellipsis
        i_tensors[k] = tensor.align_to(*k_dims)#, ...)
    
    # TODO: Can just pop the first i_tensor?
    T = 1 

    # 5. multiply
    for k, tensor in i_tensors.items() :
        T = T * tensor

    # 6. sum out dim
    T = T.sum(dim)
    
    # 7. put it back
    other_tensors[dim] = T
    
    return other_tensors


def reduce_by_dim(d, dim):
    assert( tra.k_dim_name("") in dim )
    
    other_tensors, i_tensors = get_dependent_factors(d, dim)
    all_names = get_all_names(i_tensors)
    k_dims = filter_names(all_names, plates=False)
    
    # 4. use torch names to get all their dims in same order
    for d, tensor in i_tensors.items() :
        i_tensors[d] = tensor.align_to(*k_dims, ...)
    
    key = list(i_tensors.keys())[0]
    T = i_tensors.pop(key) 
    
    # 5. multiply 
    for k, tensor in i_tensors.items() :
        T = u.logmulexp(tensor, T, dim)

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