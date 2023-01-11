import torch as t
import torch.nn as nn
import alan
t.manual_seed(0)

J = 2
M = 3
N = 4
platesizes = {'plate_1': J, 'plate_2': M, 'plate_3': N}

class PQ(alan.PQ):
    def __init__(self):
        super().__init__()
        self.a = alan.TiltedNormal()
        self.b = alan.TiltedNormal()
        self.c = alan.TiltedNormal({'plate_1': J})
        self.d = alan.TiltedNormal({'plate_1': J, 'plate_2': M})
    def forward(self, tr):
        tr.PQ('a',   self.a(0, 1))
        tr.PQ('b',   self.b(tr['a'], 1))
        tr.PQ('c',   self.c(tr['b'], 1, plates='plate_1'))
        tr.PQ('d',   self.d(tr['c'], 1, plates='plate_2'))
        tr.P('obs', alan.Normal(tr['d'], 0.01), plates='plate_3')
pq = PQ()

data = alan.sample_P(pq, platesizes=platesizes, varnames=('obs',))

model = alan.Model(pq, data={'obs': data['obs']})

K=100
for i in range(200):
    print(model.elbo(K).item())
    model.ml_update(K, 0.2)
