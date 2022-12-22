import math
import torch as t
import torch.distributions as td
import tpp
import tpp.postproc as pp
from testing_utils import test_vs_int
from tpp.utils import dim2named_tensor

pobs_z1 = 0.8
pobs_z0 = 0.3

N = 3
obs = (t.rand(N)>0.5).to(dtype=t.float).rename("plate_1")
obs_all = (t.rand(N+2)>0.5).to(dtype=t.float).rename("plate_1") 
obs_all[:N] = obs

def P_int(tr):
    tr.sample('theta', tpp.Beta(1, 1))
    tr.sample('obs',   tpp.Bernoulli(pobs_z1*tr['theta'] + pobs_z0*(1-tr['theta'])))

#### Approximate posterior over z

def P(tr):
    tr.sample('theta', tpp.Beta(1, 1))
    tr.sample('z',     tpp.Bernoulli(tr['theta']), plates="plate_1")
    tr.sample('obs',   tpp.Bernoulli(pobs_z1*tr['z'] + pobs_z0*(1-tr['z'])))

Q_prior = lambda tr: None
def Q_fac(tr):
    tr.sample('theta', tpp.Beta(1, 1))
    tr.sample('z',     tpp.Bernoulli(0.5), plates="plate_1")
def Q_nonfac(tr):
    tr.sample('theta', tpp.Beta(1, 1))
    tr.sample('z',     tpp.Bernoulli(tr['theta']), plates="plate_1")
Qs = [Q_prior, Q_fac, Q_nonfac]

test_vs_int(P_int, P, Qs, obs, obs_all, K=1000, N=1001)

#### Sum over z

def P(tr):
    tr.sample('theta', tpp.Beta(1, 1))
    tr.sample('z',     tpp.Bernoulli(tr['theta']), plates="plate_1", sum_discrete=True)
    tr.sample('obs',   tpp.Bernoulli(pobs_z1*tr['z'] + pobs_z0*(1-tr['z'])))

Q_prior = lambda tr: None
def Q_fac(tr):
    tr.sample('theta', tpp.Beta(1, 1))
Qs = [Q_prior, Q_fac]

test_vs_int(P_int, P, Qs, obs, obs_all, K=1000, N=1001)
