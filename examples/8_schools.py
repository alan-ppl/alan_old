import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from torch.distributions import transforms

### data
J = 8
y = {'obs': t.tensor([28, 8, -3, 7, -1, 1, 18, 12])}
sigma = t.tensor([15, 10, 16, 11, 9, 11, 10, 18])


def P(tr):
  '''
  Heirarchical model for 8 schools example
  '''
  tr['mu'] = tpp.Normal(0, 10)
  tr['tau'] = tpp.Normal(5, 1)
  tr['theta'] = tpp.Normal(tr['mu'],tr['tau'].exp(),sample_shape=J,sample_names="plate_1")
  tr['obs'] = tpp.Normal(tr['theta'],sigma, sample_shape=J, sample_names="plate_1")

class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_mu = nn.Parameter(t.randn(1))
        self.scale_mu = nn.Parameter(0.1*t.rand(1))

        self.loc_logtau = nn.Parameter(t.randn(1))
        self.scale_logtau = nn.Parameter(0.1*t.rand(1))


        self.loc_theta=nn.Parameter(t.randn((J,), names=('plate_1',)))
        self.scale_theta = nn.Parameter(0.1*t.rand((J,), names=('plate_1',)))


    def forward(self, tr):
        tr['mu'] = tpp.Normal(self.loc_mu, self.scale_mu.exp())
        tr['tau'] = tpp.Normal(self.loc_logtau, self.scale_logtau.exp())
        tr['theta'] = tpp.Normal(self.loc_theta,self.scale_theta.exp())


model = tpp.Model(P, Q(), y)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

print("K=100")
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(K=20)
    (-elbo).backward()
    opt.step()

    if 0 == i%500:
        print(elbo.item())


print(model.Q.loc_mu)
print(model.Q.scale_mu)
print(model.Q.scale_logtau.exp())
print(model.Q.loc_theta)
print(model.Q.scale_theta.exp())
