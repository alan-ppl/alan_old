import torch as t
import torch.nn as nn
import tpp
from tpp.backend import vi
import tqdm
from functorch.dim import dims

'''
Test posterior inference with a Gaussian with plated observations
'''
plate_1 = dims(1 , [10])
def P(tr):
  '''
  Bayesian Heirarchical Gaussian Model
  '''
  a = t.zeros(5,)
  tr['mu'] = tpp.MultivariateNormal(a, t.eye(5))
  tr['obs'] = tpp.MultivariateNormal(tr['mu'], t.eye(5), sample_dim=plate_1)



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(5,))


        self.log_s_mu = nn.Parameter(t.zeros(5,))


    def forward(self, tr):
        tr['mu'] = tpp.MultivariateNormal(self.m_mu, t.diag(self.log_s_mu.exp()))

data = tpp.sample(P, 'obs')



model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K = 5
dim = tpp.make_dims(P, K, [plate_1])

for i in range(1):
    opt.zero_grad()
    elbo = model.liw(dims=dim)
    (-elbo).backward()
    opt.step()
