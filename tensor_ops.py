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
        dim = k_dims.pop(random.randrange(len(k_dims)))
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


def naive_reduce_by_dim(d, dim):
    other_tensors, i_tensors = get_dependent_factors(d, dim)
    all_names = get_all_names(i_tensors)
    k_dims = filter_names(all_names, plates=False)
    
    # 4. use torch names to get all their dims in same order
    for k, tensor in i_tensors.items() :
        i_tensors[k] = tensor.align_to(*k_dims, ...)
    
    # TODO: "this can just be a scalar", T=1
    # T = 1
    s = get_max_shape(i_tensors)
    T = t.ones(s)

    # 5. multiply them, as in `*`
    for k, tensor in i_tensors.items() :
        T *= tensor

    # 6. sum out `__a`
    T = T.sum(dim)
    print(T)
    other_tensors[dim] = T
    
    return other_tensors


def reduce_by_dim(d, dim):
    other_tensors, i_tensors = get_dependent_factors(d, dim)
    all_names = get_all_names(i_tensors)
    k_dims = filter_names(all_names, plates=False)
    
    # 4. use torch names to get all their dims in same order
    for k, tensor in i_tensors.items() :
        i_tensors[k] = tensor.align_to(*k_dims, ...)
    
    s = get_max_shape(i_tensors)
    T = t.ones(s)
    
    # 5. multiply them
    for k, tensor in i_tensors.items() :
        T = u.logmeanexp(tensor, T)

    # 6. sum out `__a`
    T = T.sum(dim)
    print(T)
    other_tensors[dim] = T
    
    return other_tensors


# factors T∣K_i that depend on K_i
def get_dependent_factors(d, i):
    tensors = d.values()
    
    # TODO: consider lists instead
    dependents = { k: tensor for k, tensor in d.items() \
                    if i in tensor.names }
    nondependents = { k: tensor for k, tensor in d.items() \
                        if i not in tensor.names }
    
    return nondependents, dependents


# Dreadful hack
def get_max_shape(tensors) :
    shapes = [list(t.shape) for _, t in tensors.items()]
    size = max([max(shape) for shape in shapes])
    
    return [size] * len(shapes[0]) 


if __name__ == "__main__" :
    kappa = 2
    n = 2
    data = {} # {"a": [4] * 100}
    tr = sample_and_eval(chain_dist, draws=kappa, nProtected=n, data=data)
    tensors = tr.trace.out_dicts['log_prob']

    print(tpp.combine_tensors(tensors))