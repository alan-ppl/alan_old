import torch as t
from .backend import *


def vi(logps, logqs, dims):
    elbo, _ = sum_logpqs(logps, logqs, dims)
    return elbo

def reweighted_wake_sleep(logps, logqs, dims):

    # ## Wake-phase Theta p update
    wake_theta_loss, marginals = sum_logpqs(logps, {n:lq.detach() for (n,lq) in logqs.items()}, dims)
    # print(wake_theta_loss)
    ## Wake-phase phi q update
    logps = {n:lp.detach() for (n,lp) in logps.items()}
    wake_phi_loss, marginals = sum_logpqs(logps, logqs, dims)
    # print(wake_phi_loss)
    ## Sleep-phase phi q update

    return wake_theta_loss, wake_phi_loss


## TODO: figure out how to ensure samples are right shape
## (right now the shapes corresponding to the plates can be transposed)
## Idea: get list of plate dimensions and order corresponding to that
def gibbs(marginals):

    #names of random variables that we've sampled
    K_names = []
    #corresponding sampled indexes
    ks = []

    K_dict = {}
    for (rv, log_margs) in marginals[::-1]:

        #throw away log_margs without dimension of interest
        K_name = rv
        log_margs = [lm for lm in log_margs if (K_name in lm.names)]


        #index into log_margs with previously sampled ks
        selected_lms = []
        # after first K_name sampled for
        ## TODO do we need outer if statement??
        if len(K_dict.keys()) > 0:
            for name in K_dict.keys():
                # If sampled k from name corresponds to a plate, i.e log marg will have an additional
                # dimension corresponding to the plate sample size
                if K_dict[name].dim() > 0:
                    # append each log marginal from each index corresponding to a different sample in the plate
                    for index in K_dict[name].tolist():
                        selected_lms.extend([lm.select(name, index) for lm in log_margs if name in lm.names])
                else: # only one index in k, i.e doesn't come from a plate
                    selected_lms.extend([lm.select(name, K_dict[name]) for lm in log_margs if name in lm.names])
        else: # first K_name sampled for
            selected_lms = [lm.align_to(*K_names, '...')[tuple(ks)] for lm in log_margs]


        log_margs = selected_lms

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
        ks.append(td.Categorical(logits=lp.rename(None)).sample())
        K_dict[K_names[-1]] = ks[-1]

    return K_dict

# def gibbs(marginals):
#
#     #names of random variables that we've sampled
#     K_names = []
#     #corresponding sampled indexes
#     ks = []
#
#     K_dict = {}
#     for (rv, log_margs) in marginals[::-1]:
#
#         #throw away log_margs without dimension of interest
#         K_name = rv
#         log_margs = [lm for lm in log_margs if (K_name in lm.names)]
#         print(log_margs)
#
#
#         #Sample K for each dimension and then use that K to pick
#         # problem is that aligning the tensors adds a dimension to the shape
#         # so that when i come to index in, im indexing in the wrong dimension
#
#         #index into log_margs with previously sampled ks
#         #different indexing behaviour for tuples vs lists
#         # print(rv)
#         # print("indexes: ")
#         # print(tuple(ks))
#         # print(K_names)
#
#
#         # log_margs = [lm.align_to(*K_names, '...')[tuple(ks)] for lm in log_margs]
#         # print([lm.names for lm in log_margs])
#         # print([lm.shape for lm in log_margs])
#         # print(K_dict)
#         #print([lm for lm in log_margs])
#         selected_lms = []
#         if len(K_dict.keys()) > 0:
#             for name in K_dict.keys():
#                 # print(K_dict[name].dim())
#                 if K_dict[name].dim() > 0:
#                     for index in K_dict[name].tolist():
#                         selected_lms.extend([lm.select(name, index) for lm in log_margs if name in lm.names])
#                         # print([lm.select(name, index) for lm in log_margs if name in lm.names])
#                 else:
#                     selected_lms.extend([lm.select(name, K_dict[name]) for lm in log_margs if name in lm.names])
#                     # print([lm.select(name, K_dict[name]) for lm in log_margs if name in lm.names])
#         else:
#             selected_lms = [lm.align_to(*K_names, '...')[tuple(ks)] for lm in log_margs]
#
#
#         log_margs = selected_lms #+ no_shared_dims
#
#         #the only K left should be K_name
#         #and plates should all be the same (and there should be more than one tensor)
#         #therefore all dim_names should be the same,
#         dmss = [set(lm.names) for lm in log_margs]
#         dms0 = dmss[0]
#
#         for dms in dmss[1:]:
#             assert dms0 == dms
#
#         #the only K left should be K_name
#         remaining_K_names = [n for n in dms0 if is_K(n)]
#         assert 1==len(remaining_K_names)
#         assert K_name == remaining_K_names[0]
#
#         #align and combine tensors
#         plate_names = [n for n in dms0 if is_plate(n)]
#         align_names = plate_names + remaining_K_names
#
#         lp = sum([lm.align_to(*align_names) for lm in log_margs])
#         print('final log prob shape')
#         print(lp)
#         print(lp.shape)
#         #add K_name and sample to lists
#         K_names.append(remaining_K_names[0])
#         ks.append(td.Categorical(logits=lp.rename(None)).sample())
#         print(ks[-1])
#         K_dict[K_names[-1]] = ks[-1]
#
#
#
#     return K_dict




if __name__ == "__main__":
    a = t.randn(3,3).refine_names('K_d', 'K_b')
    ap = t.randn(3,3).refine_names('K_b', 'K_a')
    b = t.randn(3,3,3).refine_names('K_a', 'K_b', 'plate_s')
    c = t.randn(3,3,3).refine_names('K_c', 'K_d', 'plate_s')
    d = t.randn(3,3,3).refine_names('K_a', 'K_c', 'plate_b')
    lps = (a,b,c,d)

    assert t.allclose((a.exp() @ ap.exp()/3).log().rename(None), reduce_K([a, ap], 'K_b')[0].rename(None))

    lp, marginals = sum_lps(lps)

    # data = tpp.sample(P, "obs")
    # print(data)
    print(gibbs(marginals))
