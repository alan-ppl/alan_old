import torch as t
import torch.nn as nn
import tpp
t.manual_seed(0)

def P(tr):
    tr.sample('decay', tpp.Normal(-3, 1), plate="plate_1")
    transition = lambda x: tpp.Normal(t.exp(tr['decay']) * x, 0.01)
    tr.sample('ts', tpp.Timeseries(0, transition), T="Tb")
    tr.sample('obs', tpp.Normal(tr['ts'], 1))

def Q(tr):
    tr.sample('decay', tpp.Normal(-3, 1), plate="plate_1")
    transition = lambda x: tpp.Normal(t.exp(tr['decay']) * x, 0.01)
    tr.sample('ts', tpp.Timeseries(0, transition), T="Tb")

data = tpp.sample(P, varnames=('obs',), sizes={"Tb": 20, "plate_1": 3})

model = tpp.Model(P, Q, data)
sample = model.sample(5, True, {})
elbo = model.elbo(5)
