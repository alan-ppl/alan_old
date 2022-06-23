import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm



data_x = t.tensor([[0.4,1],
                   [0.5,1],
                   [0.24,1],
                   [-0.68,1],
                   [-0.4,1],
                   [-0.3,1],
                   [0.9,1]]).t()


sigma_w = 0.5
sigma_y = 0.1
w_0 = t.randn(2,)

def P(tr):
  '''
  Bayesian Gaussian Linear Model
  '''

  tr['w'] = tpp.MultivariateNormal(w_0, sigma_w*t.eye(2))
  tr['obs'] = tpp.Normal(tr['w'] @ data_x, sigma_y**(1/2))



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(2,))


        self.s_mu = nn.Parameter(t.randn(2,2))



    def forward(self, tr):
        sigma_nn = self.s_mu @ self.s_mu.t()
        sigma_nn.add_(t.eye(2) * 1e-5)
        tr['w'] = tpp.MultivariateNormal(self.m_mu, sigma_nn)


data_y = tpp.sample(P,"obs")
print(data_y)

model = tpp.Model(P, Q(), data_y)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

print("K=5")
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(K=5)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())


# ws = []
# for i in range(1000):
#     sample = tpp.sample(model.Q)
#     ws.append(sample['w'].rename(None).flatten())
#
# print(t.round(t.cov((t.vstack(ws)).T), decimals=3))
# print(t.mean((t.vstack(ws)), dim=0))


print("Approximate mu")
print(model.Q.m_mu)

print("Approximate Covariance")
inferred_cov = model.Q.s_mu @ model.Q.s_mu.t()
inferred_cov.add_(t.eye(2)* 1e-5)
print(inferred_cov)


V_n = sigma_y * t.inverse(sigma_y * t.inverse(sigma_w*t.eye(2)) + data_x @ data_x.t())
w_n = V_n @ t.inverse(sigma_w*t.eye(2)) @ w_0.reshape(-1,1) + (1/sigma_y) * V_n @ data_x @ data_y['obs'].reshape(-1,1)


print("True mean weights")
print(w_n)

print("True weights covariance")
print(V_n)
