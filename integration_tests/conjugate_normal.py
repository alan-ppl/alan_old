import math
import torch as t
import torch.distributions as td
import alan
import alan.postproc as pp
from testing_utils import within_stderrs
from alan.utils import dim2named_tensor

s_theta = t.ones(())
s_z     = t.ones(())
s_obs   = t.ones(())
v_theta = s_theta**2
v_z     = s_z**2
v_obs   = s_obs**2

def P(tr):
    tr.sample('theta', alan.Normal(0, s_theta))
    tr.sample('z',     alan.Normal(tr['theta'], s_z), plate="plate_1")
    tr.sample('obs',   alan.Normal(tr['z'], s_obs))

def P_int(tr):
    tr.sample('theta', alan.Normal(0, s_theta))
    tr.sample('obs',   alan.Normal(tr['theta'], (v_z + v_obs).sqrt()))

N = 2
obs = t.ones(N) + 0.1 * t.randn(N)
obs = obs.rename("plate_1")

#True posterior means and variances
post_v  = 1/(1/v_theta + N/(v_obs+v_z))
post_m  = post_v * (obs.sum() / (v_obs + v_z))
post_m2 = post_m**2 + post_v

#By Bayes:
#P(z|x) = P(x, z) / P(x)
#Rearranging:
#P(x) = P(x,z) / P(z|x)
#So if we have the joint and posterior, we can get the marginal.
z = t.zeros(())
logjoint = td.Normal(0, s_theta).log_prob(z) + td.Normal(z, (v_z+v_obs).sqrt()).log_prob(obs.rename(None)).sum()
logpost = td.Normal(post_m, post_v.sqrt()).log_prob(z)
ev = logjoint - logpost

Q_prior = lambda tr: None
def Q_fac(tr):
    tr.sample('theta', alan.Normal(0, s_theta))
    tr.sample('z',     alan.Normal(0, (v_z+v_theta).sqrt()), plate="plate_1")
def Q_nonfac(tr):
    tr.sample('theta', alan.Normal(0, s_theta))
    tr.sample('z',     alan.Normal(tr['theta'], s_z), plate="plate_1")

K = 1000
N = 1001
Qs = [Q_prior, Q_fac, Q_nonfac]
for Q in Qs:
    #Model where we sample from prior
    model = alan.Model(P, Q, data={'obs':obs})

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
        elbos.append(model.elbo(K))
    elbos = t.stack(elbos, 0)
    elbo_mean = elbos.mean()
    elbo_stderr = elbos.std() / math.sqrt(elbos.shape[0])
    print(elbo_mean.item())
    within_stderrs(ev, elbo_mean, elbo_stderr)
