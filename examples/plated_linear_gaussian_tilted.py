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
p = P()
#def q(tr):
#    tr.sample('a',   tpp.Normal(t.zeros(()), 1),                )

#def p(tr):
#    tr.sample('a',   tpp.Normal(t.zeros(()), 1),                )
#    tr.sample('b',   tpp.Normal(tr['a'], 1),                    )
#    tr.sample('c',   tpp.Normal(tr['b'], 1),    plates='plate_1')
#    tr.sample('d',   tpp.Normal(tr['c'], 1),    plates='plate_2')
#    tr.sample('obs', tpp.Normal(tr['d'], 0.01), plates='plate_3')

data = tpp.sample(p, platesizes=platesizes, varnames=('obs',))

model = tpp.Model(p, data={'obs': data['obs']})

K=100
for i in range(40):
    print(model.elbo(K).item())
    model.update(K, 0.2)
