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
    tr.sample('obs', tpp.Normal(tr['d'], 1), plates='plate_3')

data = tpp.sample(P, platesizes=platesizes, N=100, varnames=('obs',))
prior_samples = tpp.sample(P, platesizes=platesizes, N=100)

dists = {'a': tpp.Normal, 'b': tpp.Normal, 'c': tpp.Normal, 'd': tpp.Normal}
Q = tpp.NatQ(prior_samples, dists, data=data)

#model = tpp.Model(P, Q(), {'obs': data['obs']})
#
#
#K=10
#for i in range(20000):
