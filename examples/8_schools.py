import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from torch.distributions import transforms
import torch.distributions as td

device = t.device("cuda" if t.cuda.is_available() else "cpu")

### data
J = t.tensor(8)
y = {'obs': t.tensor([28, 8, -3, 7, -1, 1, 18, 12])}
sigma = t.tensor([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


def P(tr):
  '''
  Heirarchical model for 8 schools example
  '''
  tr['mu'] = tpp.Normal(t.tensor(0.0), t.tensor(5.0))
  tr['tau'] = tpp.HalfCauchy(t.tensor(5.0))
  tr['theta'] = tpp.Normal(t.tensor(0.0),1,sample_shape=(1,J),sample_names="plate_1")
  tr['obs'] = tpp.Normal(tr['mu'] + tr['tau']*tr['theta'],sigma)


class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_mu = nn.Parameter(t.randn(1))
        self.scale_mu = nn.Parameter(0.1*t.rand(1))

        self.loc_logtau = nn.Parameter(t.zeros(1))
        self.scale_logtau = nn.Parameter(0.1*t.rand(1))


        self.loc_theta=nn.Parameter(t.randn((J,), names=('plate_1',)))
        self.scale_theta = nn.Parameter(0.1*t.rand((J,), names=('plate_1',)))


    def forward(self, tr):
        tr['mu'] = tpp.Normal(self.loc_mu, self.scale_mu.exp())
        tr['tau'] = tpp.LogNormal(self.loc_logtau, self.scale_logtau.exp())

        tr['theta'] = tpp.Normal(self.loc_theta,self.scale_theta.exp())


model = tpp.Model(P, Q(), y)
tpp.sample(P)
model.to(device)
opt = t.optim.Adam(model.parameters(), lr=1E-3)
print("K=10")
for i in range(10):
    opt.zero_grad()
    elbo = model.elbo(K=10)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())


print(model.Q.loc_mu)
print(model.Q.scale_mu.exp())
print((model.Q.loc_logtau + 1/2 * model.Q.scale_logtau.exp()).exp())
print(model.Q.loc_theta  + model.Q.loc_mu)
print(model.Q.scale_theta.exp())

print(tpp.sample(Q()))