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
        self.Qa = tpp.MLNormal()
        self.Qb = tpp.MLNormal()
        self.Qc = tpp.MLNormal({'plate_1': J})
        self.Qd = tpp.MLNormal({'plate_1': J, 'plate_2': M})

    def forward(self, tr):
        tr.sample('a', self.Qa())
        tr.sample('b', self.Qb())
        tr.sample('c', self.Qc())
        tr.sample('d', self.Qd())

data = tpp.sample(P, platesizes=platesizes, varnames=('obs',))

class PQ(tpp.QModule):
    def __init__(self):
        super().__init__()
        self.Qa = tpp.MLNormal()
        self.Qb = tpp.MLNormal()
        self.Qc = tpp.MLNormal({'plate_1': J})
        self.Qd = tpp.MLNormal({'plate_1': J, 'plate_2': M})
    def forward(self, tr):
        tr.sample('a',   tpp.Normal(t.zeros(()), 1),                  delayed_Q=self.Qa)
        tr.sample('b',   tpp.Normal(tr['a'], 1),                      delayed_Q=self.Qb)
        tr.sample('c',   tpp.Normal(tr['b'], 1),    plates='plate_1', delayed_Q=self.Qc)
        tr.sample('d',   tpp.Normal(tr['c'], 1),    plates='plate_2', delayed_Q=self.Qd)
        tr.sample('obs', tpp.Normal(tr['d'], 0.01), plates='plate_3')

K = 100
T = 20
lr = 0.2

t.manual_seed(0)
m1 = tpp.Model(P, Q(), data={'obs': data['obs']})
for i in range(T):
    print(m1.elbo(K).item())
    m1.update(K, lr)

print() 
print()
t.manual_seed(0)
m2 = tpp.Model(PQ(), data={'obs': data['obs']})
for i in range(T):
    print(m2.elbo(K).item())
    m2.update(K, lr)
