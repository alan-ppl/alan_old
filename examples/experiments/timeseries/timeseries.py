import torch as t
import torch.nn as nn
import alan
import numpy as np
t.manual_seed(0)

N = 2
def P(tr):
    transition = lambda x: alan.Normal(x, 1/np.sqrt(N))
    tr.sample('ts', alan.Timeseries(0, transition), T="Tb")
    tr.sample('obs', alan.Normal(tr['ts'], 1))

def Q(tr):
    transition = lambda x: alan.Normal(x, 1/np.sqrt(N))
    tr.sample('ts', alan.Timeseries(0, transition), T="Tb")

data = alan.sample(P, varnames=('obs',), platesizes={"Tb": N})

model = alan.Model(P, Q, data)

print(model.elbo(2))
# print(model.elbo(3))
# print(model.elbo(10))
# print(model.elbo(30))

t.manual_seed(0)

def P(tr):
    transition = lambda x: alan.Normal(x, 1/np.sqrt(N))
    tr.sample('ts', alan.Timeseries(0, transition), T="Tb")
    tr.sample('obs', alan.Normal(tr['ts'], 1))

def Q(tr):
    transition = lambda x: alan.Normal(x, 1/np.sqrt(N))
    tr.sample('ts', alan.Timeseries(0, transition), T="Tb")

data = alan.sample(P, varnames=('obs',), platesizes={"Tb": N})

model = alan.Model(P, Q, data)

print(model.elbo_tmc(2))
# print(model.elbo_tmc(3))
# print(model.elbo_tmc(10))
# print(model.elbo_tmc(30))
