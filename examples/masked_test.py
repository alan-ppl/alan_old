import torch as t
import torch.nn as nn
import alan
t.manual_seed(0)

J = 2

platesizes = {'plate_1': J}
def P(tr):
  '''
  Bayesian Gaussian Model
  '''
  a = t.zeros(5)
  tr('mu', alan.Normal(a, tr.ones(5)) , plates="plate_1")
  tr('obs', alan.Normal(tr['mu'], tr.ones(5)))



class Q(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(5,))
        self.log_s_mu = nn.Parameter(t.zeros(5,))

    def forward(self, tr):
        tr('mu', alan.Normal(self.m_mu, self.log_s_mu.exp()))


data = alan.Model(P).sample_prior(platesizes=platesizes, varnames='obs')

mask = {'obs': t.FloatTensor(data['obs'].shape).uniform_() > 0.8}
mask = {}
print(data['obs'])
print(mask)
K = 100
T = 40
lr = 0.1

t.manual_seed(0)
q = Q()
m1 = alan.Model(P, q).condition(data=data, mask=mask)
for i in range(T):
    sample = m1.sample_same(K, reparam=False)
    print(sample.elbo().item())
    m1.update(lr, sample)