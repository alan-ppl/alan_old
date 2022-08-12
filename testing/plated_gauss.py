import torch as t
import torch.nn as nn
import tpp
from tpp.backend import vi
import tqdm
from torchdim import dims

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

data = {'obs': t.tensor([[ 0.1778, -0.4364, -0.2242, -2.0126, -0.4414],
[ 2.4132,  0.1589, -0.1721,  1.4035,  0.5189],
[ 1.9956, -0.6153,  0.3413, -0.9390, -1.6557],
[ 0.7620,  0.1262, -0.3963,  2.6029,  0.2131],
[ 1.1981, -0.8900,  0.7388,  0.1689, -1.3313],
[ 1.7920, -0.4034,  1.1757,  0.4693, -0.5890],
[ 0.5391,  0.4714,  0.5067,  1.2729,  0.9414],
[ 1.4357,  0.0208,  0.7751,  1.5554,  0.8555],
[ 0.1909, -0.3226,  0.5594,  1.0569, -1.6546],
[-0.1745, -1.9498,  1.5145,  2.7684, -0.8587]])}



model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K = 5
dim = tpp.make_dims(P, K, [plate_1])

for i in range(15000):
    opt.zero_grad()
    elbo = model.elbo(dims=dim)
    (-elbo).backward()
    opt.step()

inferred_mean = model.Q.m_mu

inferred_cov = model.Q.log_s_mu.exp()

true_mean = t.mm(t.inverse(t.eye(5) + 1/10 * t.eye(5)),data['obs'].mean(axis=0).reshape(-1,1))
true_cov = t.inverse(t.eye(5) + (1/10 * t.eye(5))) * 1/10

print(true_mean)
print(inferred_mean)
print(true_cov)
print(t.diag(inferred_cov))
assert((t.abs(true_mean - inferred_mean.reshape(-1,1))<0.05).all())
assert((t.abs(true_cov - t.diag(inferred_cov))<0.05).all())
