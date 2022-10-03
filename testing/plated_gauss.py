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

for i in range(15000):
    opt.zero_grad()
    elbo = model.elbo(dims=dim)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())
        
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
