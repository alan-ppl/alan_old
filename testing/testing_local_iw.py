import torch as t
import torch.nn as nn
import tpp
from tpp.backend import vi
import tqdm
from functorch.dim import dims

'''
Test posterior inference with a Gaussian with plated observations
'''
plate_1,plate_2, plate_3 = dims(3 , [5,10,15])
def P(tr):
  '''
  Bayesian Heirarchical Gaussian Model
  '''
  tr['mu'] = tpp.Normal(t.zeros(1,), t.ones(1,))

  tr['psi'] = tpp.Normal(tr['mu'], t.ones(1,), sample_dim=plate_1, group='local')

  tr['phi'] = tpp.Normal(tr['psi'], t.ones(1,), sample_dim=plate_2, group='local')

  tr['obs'] = tpp.Normal(tr['phi'], t.ones(5,), sample_dim=plate_3)



class Q(tpp.Q_module):
    def __init__(self):
        super().__init__()
        self.reg_param("m_mu", t.zeros((1,)))
        self.reg_param("log_s_mu", t.zeros((1,)))

        self.reg_param("m_psi", t.zeros((5,)), [plate_1])
        self.reg_param('log_s_psi', t.zeros((5,)), [plate_1])


        self.reg_param('m_phi', t.zeros((5,10)), [plate_1,plate_2])
        self.reg_param('log_s_phi', t.zeros((5,10)), [plate_1,plate_2])


    def forward(self, tr):
        tr['mu'] = tpp.Normal(self.m_mu, self.log_s_mu.exp(), sample_K=False)
        tr['psi'] = tpp.Normal(self.m_psi, self.log_s_psi.exp())
        tr['phi'] = tpp.Normal(self.m_phi, self.log_s_phi.exp())

data = tpp.sample(P, 'obs')
test_data = tpp.sample(P, 'obs')
# print(tpp.sample(P))


model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K = 5
dim = tpp.make_dims(P, K)

for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(dims=dim)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())
