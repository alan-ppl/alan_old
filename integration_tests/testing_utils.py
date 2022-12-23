import math
import torch as t

import tpp
import tpp.postproc as pp
from tpp.utils import dim2named_tensor

def within_stderrs(true_value, est, stderr):
    sigmas = 6
    assert (est - sigmas*stderr < true_value) and (true_value < est + sigmas*stderr)


def test_vs_int(P_int, P, Qs, obs, obs_all, K=1000, N=1001):
    int_model = tpp.Model(P_int, data={'obs':obs})
    int_weights = int_model.weights(1000)
    post_v  = pp.var(int_weights)['theta']
    post_m  = pp.mean(int_weights)['theta']
    post_m2 = pp.mean(pp.square(int_weights))['theta']

    ev = int_model.elbo(1000, reparam=False)

    for Q in Qs:
        #Model where we sample from prior
        model = tpp.Model(P, Q, data={'obs':obs})

        #Check E[theta] and E[theta**2] are within stderrs, using importance weighting
        weights = model.weights(K)

        iw_m  = pp.mean(weights)['theta']
        iw_m2 = pp.mean(pp.square(weights))['theta']

        se_m  = pp.stderr(weights)['theta']
        se_m2 = pp.stderr(pp.square(weights))['theta']

        within_stderrs(post_m, iw_m, se_m)
        within_stderrs(post_m2, iw_m2, se_m2)

        #Check E[theta] and E[theta**2] are within stderrs, using importance samples
        samples = dim2named_tensor(model.importance_samples(K, N)['theta'])

        is_m  = samples.mean(0)
        is_m2 = (samples**2).mean(0)

        within_stderrs(post_m, is_m, se_m)
        within_stderrs(post_m2, is_m2, se_m2)

        #Check model evidence
        elbos = []
        for _ in range(100):
            elbos.append(model.elbo(K, reparam=False))
        elbos = t.stack(elbos, 0)
        elbo_mean = elbos.mean()
        elbo_stderr = 4*elbos.std() / math.sqrt(elbos.shape[0])
        #print(elbo_mean.item())
        within_stderrs(ev, elbo_mean, elbo_stderr)

        print()
        print(Q)
        print(model.predictive_ll(K, N, data_all={'obs':obs_all}))
