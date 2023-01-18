import torch as t
import torch.nn as nn
import alan
t.manual_seed(0)

def P(tr):
    tr.sample('decay', alan.Normal(-3, 1), plates="plate_1")
    transition = lambda x: alan.Normal(t.exp(tr['decay']) * x, 0.01)
    tr.sample('ts', alan.Timeseries(0, transition), T="Tb")
    tr.sample('obs', alan.Normal(tr['ts'], 1))

def Q(tr):
    tr.sample('decay', alan.Normal(-3, 1), plates="plate_1")
    transition = lambda x: alan.Normal(t.exp(tr['decay']) * x, 0.01)
    tr.sample('ts', alan.Timeseries(0, transition), T="Tb")

data = alan.sample(P, varnames=('obs',), platesizes={"Tb": 20, "plate_1": 3})

model = alan.Model(P, Q, data)

elbo = model.elbo(5)


obs = t.randn((30, 4), names=('Tb', 'plate_1'))
obs[:20, :3] = data['obs']
pred_ll = model.predictive_ll(5, 10, data_all={"obs": obs})
print(pred_ll)
