import math
import torch as t

import alan
import alan.postproc as pp
from alan.utils import dim2named_tensor

def within_stderrs(true_value, est, stderr):
    sigmas = 6
    assert (est - sigmas*stderr < true_value) and (true_value < est + sigmas*stderr)


def test_vs_int(P_int, P, Qs, obs, obs_all, K=1000, N=1001):
    int_model = alan.Model(P_int).condition(data={'obs':obs})
    sample = int_model.sample_same(K=K, reparam=False)
    int_weights = sample.weights()
    post_v  = pp.var(int_weights)['theta']
    post_m  = pp.mean(int_weights)['theta']
    post_m2 = pp.mean(pp.square(int_weights))['theta']

    ev = sample.elbo()

    for Q in Qs:
        #Model where we sample from prior
        model = alan.Model(P, Q).condition(data={'obs':obs})
        sample = model.sample_same(K=K, reparam=False)

        #Check E[theta] and E[theta**2] are within stderrs, using importance weighting
        weights = sample.weights()

        iw_m  = pp.mean(weights)['theta']
        iw_m2 = pp.mean(pp.square(weights))['theta']

        se_m  = pp.stderr(weights)['theta']
        se_m2 = pp.stderr(pp.square(weights))['theta']

        within_stderrs(post_m, iw_m, se_m)
        within_stderrs(post_m2, iw_m2, se_m2)

        #Check E[theta] and E[theta**2] are within stderrs, using importance samples
        samples = dim2named_tensor(sample.importance_samples(N)['theta'])

        is_m  = samples.mean(0)
        is_m2 = (samples**2).mean(0)

        within_stderrs(post_m, is_m, se_m)
        within_stderrs(post_m2, is_m2, se_m2)

        #Check model evidence
        elbos = []
        for _ in range(100):
            elbos.append(model.sample_same(K, reparam=False).elbo())
        elbos = t.stack(elbos, 0)
        elbo_mean = elbos.mean()
        elbo_stderr = 4*elbos.std() / math.sqrt(elbos.shape[0])
        #print(elbo_mean.item())
        within_stderrs(ev, elbo_mean, elbo_stderr)

        print()
        print(Q)
        print(model.predictive_ll(sample, N, data_all={'obs':obs_all}))
