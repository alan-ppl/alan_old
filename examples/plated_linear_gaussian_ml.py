import torch as t
import torch.nn as nn
import alan
t.manual_seed(0)

J = 2
M = 3
N = 4
platesizes = {'plate_1': J, 'plate_2': M, 'plate_3': N}
def P(tr):
    tr('a',   alan.Normal(t.zeros(()), 1))
    tr('b',   alan.Normal(tr['a'], 1))
    tr('c',   alan.Normal(tr['b'], 1), plates='plate_1')
    tr('d',   alan.Normal(tr['c'], 1), plates='plate_2')
    tr('obs', alan.Normal(tr['d'], 0.01), plates='plate_3')

class Q(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.PQa = alan.MLNormal()
        self.PQb = alan.MLNormal()
        self.PQc = alan.MLNormal({'plate_1': J})
        self.PQd = alan.MLNormal({'plate_1': J, 'plate_2': M})

    def forward(self, tr):
        tr('a', self.PQa())
        tr('b', self.PQb())
        tr('c', self.PQc())
        tr('d', self.PQd())

data = alan.sample(P, platesizes=platesizes, varnames=('obs',))

#class PQ(alan.QModule):
#    def __init__(self):
#        super().__init__()
#        self.Qa = alan.MLNormal()
#        self.Qb = alan.MLNormal()
#        self.Qc = alan.MLNormal({'plate_1': J})
#        self.Qd = alan.MLNormal({'plate_1': J, 'plate_2': M})
#    def forward(self, tr):
#        tr.sample('a',   alan.Normal(t.zeros(()), 1),                  delayed_Q=self.Qa)
#        tr.sample('b',   alan.Normal(tr['a'], 1),                      delayed_Q=self.Qb)
#        tr.sample('c',   alan.Normal(tr['b'], 1),    plates='plate_1', delayed_Q=self.Qc)
#        tr.sample('d',   alan.Normal(tr['c'], 1),    plates='plate_2', delayed_Q=self.Qd)
#        tr.sample('obs', alan.Normal(tr['d'], 0.01), plates='plate_3')

K = 100
T = 20
lr = 0.2

t.manual_seed(0)
q = Q()
m1 = alan.Model(P, q).condition(data=data)
for i in range(T):
    sample = m1.sample_mp(K)
    print(sample.elbo().item())
    m1.update(lr, sample)

#print() 
#print()
#t.manual_seed(0)
#m2 = alan.Model(PQ(), data=data)
#for i in range(T):
#    print(m2.kl(K).item())
#    m2.update(K, lr)
