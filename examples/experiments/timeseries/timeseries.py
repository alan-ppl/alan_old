import torch as t
import torch.nn as nn
import alan
t.manual_seed(0)

def P(tr):
    transition = lambda x, t: alan.Normal(x, 1/(t+1))
    tr.sample('ts', alan.Timeseries(0, transition, depend_on_t=True), T="Tb")
    tr.sample('obs', alan.Normal(tr['ts'], 1))

def Q(tr):
    transition = lambda x, t: alan.Normal(x, 1/(t+1))
    tr.sample('ts', alan.Timeseries(0, transition, depend_on_t=True), T="Tb")

data = alan.sample(P, varnames=('obs',), platesizes={"Tb": 20, "plate_1": 3})

model = alan.Model(P, Q, data)
# sample = model._sample(5, True, {}, {})
# elbo = model.elbo(5)
# sample = model.importance_samples(5, 10)
#
# obs = t.randn((30, 4), names=('Tb', 'plate_1'))
# obs[:20, :3] = data['obs']
# pred_samples = model.predictive_samples(5, 10, sizes_all={'Tb': 30, 'plate_1':4})
# pred_ll = model.predictive_ll(5, 10, data_all={"obs": obs})
# print(pred_ll)
