import torch as t
import torch.nn as nn
import tpp
t.manual_seed(0)

J = 2
M = 3
N = 4
platesizes = {'plate_1': J, 'plate_2': M, 'plate_3': N}

class P(tpp.QModule):
    def __init__(self):
        super().__init__()
        self.Qa = tpp.TiltedNormal()
        self.Qb = tpp.TiltedNormal()
        self.Qc = tpp.TiltedNormal({'plate_1': J})
        self.Qd = tpp.TiltedNormal({'plate_1': J, 'plate_2': M})
    def forward(self, tr):
        tr.sample('a',   tpp.Normal(t.zeros(()), 1),                  delayed_Q=self.Qa)
        tr.sample('b',   tpp.Normal(tr['a'], 1),                      delayed_Q=self.Qb)
        tr.sample('c',   tpp.Normal(tr['b'], 1),    plates='plate_1', delayed_Q=self.Qc)
        tr.sample('d',   tpp.Normal(tr['c'], 1),    plates='plate_2', delayed_Q=self.Qd)
        tr.sample('obs', tpp.Normal(tr['d'], 0.01), plates='plate_3')
p = P()

data = tpp.sample(p, platesizes=platesizes, varnames=('obs',))

model = tpp.Model(p, data={'obs': data['obs']})

K=100
for i in range(200):
    print(model.elbo(K).item())
    model.ml_update(K, 0.2)
