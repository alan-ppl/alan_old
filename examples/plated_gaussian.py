import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm

def P(tr):
  '''
  Bayesian Heirarchical Model
  Gaussian with Wishart Prior on precision
  '''
  a = t.zeros(5,)
  tr['mu'] = tpp.MultivariateNormal(a, t.eye(5))
  tr['obs'] = tpp.MultivariateNormal(tr['mu'], t.eye(5), sample_shape=5, sample_names='plate_1')



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(5,))
        #self.m_mu = nn.Parameter(t.tensor([[-0.2655, -0.5713, -0.1689,  0.4844, -1.0193]]))


        self.log_s_mu = nn.Parameter(t.zeros(5,))
        #self.log_s_mu = nn.Parameter(t.tensor([[0.1667, 0.1667, 0.1667, 0.1667, -0.1667]]))



    def forward(self, tr):
        tr['mu'] = tpp.MultivariateNormal(self.m_mu, t.diag(self.log_s_mu.exp()))

data = tpp.sample(P, "obs")
data = {'obs': t.tensor([[[ 0.5851,  0.8783, -0.4380, -1.3839,  0.9538]],

        [[ 0.3030,  0.8338, -2.2067, -1.8815,  3.3449]],

        [[ 1.8357, -0.3146,  0.5771, -1.4885, -0.3881]],

        [[-1.0334, -0.2395,  0.3544, -2.0973,  1.8586]],

        [[ 0.5752, -0.9763, -1.0950, -0.2201,  0.4888]]],
       names=('plate_1', None, None))}


print(data)
model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

print("K=20")
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(K=20)
    (-elbo).backward()
    opt.step()

    if 0 == i%500:
        print(elbo.item())


print("Approximate mu")
print(model.Q.m_mu)

print("Approximate Covariance")
print(model.Q.log_s_mu.exp())

b_n = t.mm(t.inverse(t.eye(5) + 1/5 * t.eye(5)),data['obs'].rename(None).mean(axis=0).reshape(-1,1))
A_n = t.inverse(t.eye(5) + 1/5 * t.eye(5)) * 1/5

print("True mu")
print(b_n)

print("True covariance")
print(t.diag(A_n))
