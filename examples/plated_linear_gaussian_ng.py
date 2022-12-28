import torch as t
import torch.nn as nn
import tpp
t.manual_seed(0)

J = 2
M = 3
N = 4
platesizes = {'plate_1': J, 'plate_2': M, 'plate_3': N}
def P(tr):
    tr.sample('a',   tpp.Normal(t.zeros(()), 1))
    tr.sample('b',   tpp.Normal(tr['a'], 1))
    tr.sample('c',   tpp.Normal(tr['b'], 1), plates='plate_1')
    tr.sample('d',   tpp.Normal(tr['c'], 1), plates='plate_2')
    tr.sample('obs', tpp.Normal(tr['d'], 0.01), plates='plate_3')

class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.Qa = tpp.NGNormal()
        self.Qb = tpp.NGNormal()
        self.Qc = tpp.NGNormal({'plate_1': J})
        self.Qd = tpp.NGNormal({'plate_1': J, 'plate_2': M})

    def forward(self, tr):
        tr.sample('a', self.Qa())
        tr.sample('b', self.Qb())
        tr.sample('c', self.Qc())
        tr.sample('d', self.Qd())

data = tpp.sample(P, platesizes=platesizes, varnames=('obs',))
prior_samples = tpp.sample(P, platesizes=platesizes, N=100)

q = Q()

model = tpp.Model(P, q, {'obs': data['obs']})

K=100
for i in range(40):
    print(model.elbo(K).item())
    model.ng_update(K, 0.2)
