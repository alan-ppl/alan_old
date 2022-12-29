import torch as t
import torch.nn as nn
import alan
t.manual_seed(0)

J = 2
M = 3
N = 4
platesizes = {'plate_1': J, 'plate_2': M, 'plate_3': N}
def P(tr):
    tr.sample('a',   alan.Normal(t.zeros(()), 1))
    tr.sample('b',   alan.Normal(tr['a'], 1))
    tr.sample('c',   alan.Normal(tr['b'], 1), plates='plate_1')
    tr.sample('d',   alan.Normal(tr['c'], 1), plates='plate_2')
    tr.sample('obs', alan.Normal(tr['d'], 0.01), plates='plate_3')

class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.Qa = alan.ML2Normal()
        self.Qb = alan.ML2Normal()
        self.Qc = alan.ML2Normal({'plate_1': J})
        self.Qd = alan.ML2Normal({'plate_1': J, 'plate_2': M})

    def forward(self, tr):
        tr.sample('a', self.Qa())
        tr.sample('b', self.Qb())
        tr.sample('c', self.Qc())
        tr.sample('d', self.Qd())

data = alan.sample(P, platesizes=platesizes, varnames=('obs',))
prior_samples = alan.sample(P, platesizes=platesizes, N=100)

q = Q()

model = alan.Model(P, q, {'obs': data['obs']})

K=100
for i in range(40):
    print(model.elbo(K).item())
    model.ml_update(K, 0.2)
