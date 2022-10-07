import torch as t
import torch.nn as nn
import tpp
from tpp.backend import vi
import tqdm
from functorch.dim import dims

'''
Test posterior inference with a Gaussian with plated observations
'''
plate_1,plate_2 = dims(2 , [5,10])
def P(tr):
  '''
  Bayesian Heirarchical Gaussian Model
  '''
  tr['mu'] = tpp.Normal(t.zeros(1,), t.ones(1,), sample_K=False)

  tr['phi'] = tpp.Normal(tr['mu'], t.ones(1,), sample_dim=plate_1)

  tr['obs'] = tpp.Normal(tr['phi'], t.ones(5,), sample_dim=plate_2)



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(1,))
        self.log_s_mu = nn.Parameter(t.zeros(1,))

        self.m_phi = nn.Parameter(t.zeros(5,))
        self.log_s_phi = nn.Parameter(t.zeros(5,))


    def forward(self, tr):
        tr['mu'] = tpp.Normal(self.m_mu, self.log_s_mu.exp(), sample_K=False)
        tr['phi'] = tpp.Normal(self.m_phi, self.log_s_phi.exp())

data = tpp.sample(P, 'obs')

# print(tpp.sample(P))


model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K = 5
dim = tpp.make_dims(P, K, [plate_1], exclude=['mu'])
for i in range(1):
    opt.zero_grad()
    elbo = model.elbo(dims=dim)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())
