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
        self.Na = alan.MLNormal()
        self.Nb = alan.MLNormal()
        self.Nc = alan.MLNormal({'plate_1': J})
        self.Nd = alan.MLNormal({'plate_1': J, 'plate_2': M})

    def forward(self, tr):
        self.Na(tr, 'a')
        self.Nb(tr, 'b')
        self.Nc(tr, 'c')
        self.Nd(tr, 'd')
        

#class PQ(alan.AlanModule):
#    def __init__(self):
#        super().__init__()
#        self.Na = alan.MLNormal()
#        self.Nb = alan.MLNormal()
#        self.Nc = alan.MLNormal({'plate_1': J})
#        self.Nd = alan.MLNormal({'plate_1': J, 'plate_2': M})
#
#    def forward(self, tr):
#        tr('a',   self.Na.Q(0., 1))
#        tr('b',   self.Nb(tr['a'], 1))
#        tr('c',   self.Nc(tr['b'], 1), plates='plate_1')
#        tr('d',   self.Nd(tr['c'], 1), plates='plate_2')
#        tr('obs', alan.Normal(tr['d'], 0.01), plates='plate_3')

data = alan.Model(P).sample_prior(platesizes=platesizes, varnames='obs')

K = 100
T = 40
lr = 0.1

t.manual_seed(0)
q = Q()
m1 = alan.Model(P, q).condition(data=data)
for i in range(T):
    sample = m1.sample_same(K, reparam=False)
    print(sample.elbo().item())
    m1.update(lr, sample)

#print() 
#print()
#t.manual_seed(0)
#m2 = alan.Model(PQ()).condition(data=data)
#for i in range(T):
#    sample = m2.sample_mp(K, reparam=False)
#    print(sample.elbo().item())
#    m2.update(lr, sample)
