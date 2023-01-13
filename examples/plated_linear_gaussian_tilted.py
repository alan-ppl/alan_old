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
        self.a = alan.TiltedNormal()
        self.b = alan.TiltedNormal()
        self.c = alan.TiltedNormal({'plate_1': J})
        self.d = alan.TiltedNormal({'plate_1': J, 'plate_2': M})
    def forward(self, tr):
        tr.sample('a',   self.a(0, 1))
        tr.sample('b',   self.b(tr['a'], 1))
        tr.sample('c',   self.c(tr['b'], 1), plates='plate_1')
        tr.sample('d',   self.d(tr['c'], 1), plates='plate_2')
        tr.sample('obs', alan.Normal(tr['d'], 0.01), plates='plate_3')
p = P()

data = alan.sample(p, platesizes=platesizes, varnames=('/obs',))

model = alan.Model(p, data={'/obs': data['/obs']})

K=100
print(model.elbo(K))
for i in range(200):
    print(model.elbo(K).item())
    model.ml_update(K, 0.2)
