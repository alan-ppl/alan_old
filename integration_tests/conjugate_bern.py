import math
import torch as t
import torch.distributions as td
import tpp
import tpp.postproc as pp
from testing_utils import within_stderrs
from tpp.utils import dim2named_tensor

pobs_z1 = 0.8
pobs_z0 = 0.3

def P(tr):
    tr.sample('theta', tpp.Beta(1, 1))
    tr.sample('z',     tpp.Bernoulli(tr['theta']), plate="plate_1")
    tr.sample('obs',   tpp.Bernoulli(pobs_z1*tr['theta'] + pobs_z0*(1-tr['theta'])))

def P_int(tr):
    tr.sample('theta', tpp.Beta(1, 1))
    tr.sample('obs',   tpp.Bernoulli(pobs_z1*tr['theta'] + pobs_z0*(1-tr['theta'])))

N = 3
obs = (t.rand(3)>0.5).to(dtype=t.float).rename("plate_1")

#True posterior means and variances
int_model = tpp.Model(P_int, data={'obs':obs})
int_weights = int_model.weights(1000)
post_v  = pp.var(int_weights)['theta']
post_m  = pp.mean(int_weights)['theta']
post_m2 = pp.mean2(int_weights)['theta']

ev = int_model.elbo(1000, reparam=False)

Q_prior = lambda tr: None
def Q_fac(tr):
    tr.sample('theta', tpp.Beta(1, 1))
    tr.sample('z',     tpp.Bernoulli(0.5), plate="plate_1")
def Q_nonfac(tr):
    tr.sample('theta', tpp.Beta(1, 1))
    tr.sample('z',     tpp.Bernoulli(tr['theta']), plate="plate_1")

K = 1000
N = 1001
Qs = [Q_prior, Q_fac, Q_nonfac]
for Q in Qs:
    #Model where we sample from prior
    model = tpp.Model(P, Q, data={'obs':obs})

    #Check E[theta] and E[theta**2] are within stderrs, using importance weighting
    weights = model.weights(K)

    iw_m  = pp.mean(weights)['theta']
    iw_m2 = pp.mean2(weights)['theta']

    se_m  = pp.stderr_mean(weights)['theta']
    se_m2 = pp.stderr_mean2(weights)['theta']

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
    elbo_stderr = 2*elbos.std() / math.sqrt(elbos.shape[0])
    print(elbo_mean.item())
    within_stderrs(ev, elbo_mean, elbo_stderr)
