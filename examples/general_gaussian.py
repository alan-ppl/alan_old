import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from functorch.dim import dims

'''
Test posterior inference with a general gaussian
'''
sigma_0 = t.rand(5,5)
sigma_0 = t.mm(sigma_0, sigma_0.t())
sigma_0.add_(t.eye(5) * 1e-5)
sigma = t.rand(5,5)
sigma = t.mm(sigma, sigma.t())
sigma.add_(t.eye(5)* 1e-5)
a = t.randn(5,)

N = 10
plate_1 = dims(1 , [N])
def P(tr):
  '''
  Bayesian Gaussian Model
  '''

  tr['mu'] = tpp.MultivariateNormal(a, sigma_0)
  tr['obs'] = tpp.MultivariateNormal(tr['mu'], sigma, sample_dim=plate_1)



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(5,))

        self.s_mu = nn.Parameter(t.randn(5,5))

    def forward(self, tr):
        sigma_nn = t.mm(self.s_mu, self.s_mu.t())
        sigma_nn.add_(t.eye(5) * 1e-5)

        tr['mu'] = tpp.MultivariateNormal(self.m_mu, covariance_matrix=sigma_nn)

data = tpp.sample(P, "obs")

model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=1
dims = tpp.make_dims(P, K)
print("K={}".format(K))

for i in range(20000):
    opt.zero_grad()
    elbo = model.elbo(dims=dims)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())

inferred_mean = model.Q.m_mu
inferred_cov = t.mm(model.Q.s_mu, model.Q.s_mu.t())
inferred_cov.add_(t.eye(5)* 1e-5)

y_hat = tpp.dename(data['obs'].mean(plate_1)).reshape(-1,1)

true_cov = t.inverse(N * t.inverse(sigma) + t.inverse(sigma_0))
true_mean = (true_cov @ (N*t.inverse(sigma) @ y_hat + t.inverse(sigma_0)@a.reshape(-1,1))).reshape(1,-1)


print('True Covariance')
print(true_cov)

print('Inferred Covariance')
print(inferred_cov)

print('True Mean')
print(true_mean)

print('Inferred Mean')
print(inferred_mean)
