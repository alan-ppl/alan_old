import torch as t
import torch.nn as nn
import tpp
from tpp.backend import vi
import tqdm
from functorch.dim import dims, Dim
t.manual_seed(0)
'''
Test posterior inference with a Gaussian with plated observations
'''
plate_1,plate_2 = dims(2 , [5,10])
def P(tr):
  '''
  Bayesian Heirarchical Gaussian Model
  '''
  tr['mu'] = tpp.Normal(t.zeros(1,), t.ones(1,))
  # print('mu P')
  # print(tr['mu'])
  tr['phi'] = tpp.Normal(tr['mu'], t.ones(1,), sample_dim=plate_1)
  # print('phi P')
  # print(tr['phi'])
  tr['psi'] = tpp.Normal(t.zeros(1,), t.ones(1,))
  # print('psi P')
  # print(tr['psi'])
  tr['gamma'] = tpp.Normal(tr['psi'], t.ones(1,), sample_dim=plate_1)
  # print('gamma P')
  # print(tr['gamma'])
  tr['obs'] = tpp.Normal(tr['phi'] + tr['gamma'], t.ones(5,), sample_dim=plate_2)



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(1,))
        self.log_s_mu = nn.Parameter(t.zeros(1,))

        self.m_phi = nn.Parameter(t.zeros(5,))
        self.log_s_phi = nn.Parameter(t.zeros(5,))

        self.m_psi = nn.Parameter(t.zeros(1,))
        self.log_s_psi = nn.Parameter(t.zeros(1,))

        self.m_gamma = nn.Parameter(t.zeros(5,))
        self.log_s_gamma = nn.Parameter(t.zeros(5,))

    def forward(self, tr):
        tr['mu'] = tpp.Normal(self.m_mu, self.log_s_mu.exp())
        tr['phi'] = tpp.Normal(self.m_phi, self.log_s_phi.exp())

        tr['psi'] = tpp.Normal(self.m_psi, self.log_s_psi.exp())
        tr['gamma'] = tpp.Normal(self.m_gamma, self.log_s_gamma.exp())

data = tpp.sample(P, 'obs')
#
# print(tpp.sample(P))


model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K = 5
dim = tpp.make_dims(P, K)
K_group1 = Dim(name='K_group1', size=K)
K_group2 = Dim(name='K_group2', size=K)
K = Dim(name='K', size=K)
dim = {'K':K, 'mu':K_group2, 'phi':K_group2, 'psi': K_group1, 'gamma':K_group1}
for i in range(20000):
    opt.zero_grad()
    elbo = model.elbo(dims=dim)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())
