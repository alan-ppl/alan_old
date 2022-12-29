import torch as t
import torch.nn as nn
import alan
t.manual_seed(0)

J = 2
M = 3
N = 4
platesizes = {'plate_1': J, 'plate_2': M, 'plate_3': N}

class P(alan.QModule):
    def __init__(self):
        super().__init__()
        self.Qa = alan.TiltedNormal()
        self.Qb = alan.TiltedNormal()
        self.Qc = alan.TiltedNormal({'plate_1': J})
        self.Qd = alan.TiltedNormal({'plate_1': J, 'plate_2': M})
    def forward(self, tr):
        tr.sample('a',   alan.Normal(t.zeros(()), 1),                  delayed_Q=self.Qa)
        tr.sample('b',   alan.Normal(tr['a'], 1),                      delayed_Q=self.Qb)
        tr.sample('c',   alan.Normal(tr['b'], 1),    plates='plate_1', delayed_Q=self.Qc)
        tr.sample('d',   alan.Normal(tr['c'], 1),    plates='plate_2', delayed_Q=self.Qd)
        tr.sample('obs', alan.Normal(tr['d'], 0.01), plates='plate_3')
p = P()

data = alan.sample(p, platesizes=platesizes, varnames=('obs',))

model = alan.Model(p, data={'obs': data['obs']})

K=100
for i in range(200):
    print(model.elbo(K).item())
    model.ml_update(K, 0.2)
