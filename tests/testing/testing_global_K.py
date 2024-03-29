import torch as t
import torch.nn as nn
import alan
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
  tr['mu'] = alan.Normal(t.zeros(1,), t.ones(1,), group='group_global')
  # print('mu P')
  # print(tr['mu'])
  tr['phi'] = alan.Normal(tr['mu'], t.ones(1,), sample_dim=plate_1, group='group1')
  # print('phi P')

  tr['psi'] = alan.Normal(t.zeros(1,), t.ones(1,), group='group_global')
  # print('psi P')
  # print(tr['psi'])
  tr['gamma'] = alan.Normal(tr['psi'], t.ones(1,), sample_dim=plate_1, group='group1')
  # print('gamma P'
  tr['obs'] = alan.Normal(tr['phi'] + tr['gamma'], t.ones(5,), sample_dim=plate_2, group='group2')



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
        tr['mu'] = alan.Normal(self.m_mu, self.log_s_mu.exp())
        tr['phi'] = alan.Normal(self.m_phi, self.log_s_phi.exp())

        tr['psi'] = alan.Normal(self.m_psi, self.log_s_psi.exp())
        tr['gamma'] = alan.Normal(self.m_gamma, self.log_s_gamma.exp())

data = alan.sample(P, 'obs')
test_data = alan.sample(P, 'obs')
# print(alan.sample(P))


model = alan.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K = 5

for i in range(1000):
    opt.zero_grad()
    elbo = model.elbo(K=K)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())

dim = alan.make_dims(P, 1)

print('Test Data')
print(alan.dename(test_data['obs']).shape)
pred_lik = model.pred_likelihood(dims=dim, test_data=test_data, num_samples=1)
print(alan.dename(pred_lik).shape)
print(pred_lik)
