import torch as t
import torch.nn as nn
import alan
t.manual_seed(0)

J = 2
M = 3
N = 4
platesizes = {'plate_1': J, 'plate_2': M, 'plate_3': N}

class Mod(alan.Model):
    def __init__(self):
        super().__init__()
        self.a = alan.TiltedNormal()
        self.b = alan.TiltedNormal()
        self.c = alan.TiltedNormal({'plate_1': J})
        self.d = alan.TiltedNormal({'plate_1': J, 'plate_2': M})

    def P(self, tr):
        tr('a',   alan.Normal(0, 1))
        tr('b',   alan.Normal(tr['a'], 1))
        tr('c',   alan.Normal(tr['b'], 1), plates='plate_1')
        tr('d',   alan.Normal(tr['c'], 1), plates='plate_2')
        tr('obs', alan.Normal(tr['d'], 0.01), plates='plate_3')

    def Q(self, tr):
        tr('a',   self.a(0, 1))
        tr('b',   self.b(tr['a'], 1))
        tr('c',   self.c(tr['b'], 1), plates='plate_1')
        tr('d',   self.d(tr['c'], 1), plates='plate_2')

mod = Mod()
data = mod.sample_prior(platesizes=platesizes, varnames='obs')
cond_model = mod.condition(data={'obs': data['obs']})

K=100
print(cond_model.sample_same(K).elbo())
for i in range(40):
    sample = cond_model.sample_same(K, reparam=False)
    print(sample.elbo().item())
    cond_model.update(0.1, sample)
