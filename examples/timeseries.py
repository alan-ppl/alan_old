import torch as t
import torch.nn as nn
import alan
t.manual_seed(0)

def P(tr):
    tr('decay', alan.Normal(-3, 1), plates="plate_1")
    tr('init',  alan.Normal(0, 1), plates="plate_1")
    transition = lambda x: alan.Normal(t.exp(tr['decay']) * x, 0.01)
    tr('ts', alan.Timeseries('init', transition), T="Tb")
    tr('obs', alan.Normal(tr['ts'], 1))

def Q(tr):
    tr('decay', alan.Normal(-3, 1), plates="plate_1")
    tr('init',  alan.Normal(0, 1), plates="plate_1")
    transition = lambda x: alan.Normal(t.exp(tr['decay']) * x, 0.01)
    tr('ts', alan.Timeseries('init', transition), T="Tb")


model = alan.Model(P, Q)
data = model.sample_prior(varnames=('obs',), platesizes={"Tb": 20, "plate_1": 3})
model = model.condition(data=data)
sample = model.sample_cat(5, True)
elbo = sample.elbo()
ims = sample.importance_samples(10)

obs = t.randn((30, 4), names=('Tb', 'plate_1'))
obs[:20, :3] = data['obs']
pred_samples = model.predictive_samples(sample, 10, platesizes_all={'Tb': 30, 'plate_1':4})
pred_ll = model.predictive_ll(5, 10, data_all={"obs": obs})
print(pred_ll)
