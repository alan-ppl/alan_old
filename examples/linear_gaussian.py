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

data_y = {'obs': t.tensor([ 0.9004, -3.7564,  0.4881, -1.1412,  0.2087,0.478,-1.1])}
sigma_w = 0.5
sigma_y = 0.1
a = t.randn(2,)

def P(tr):
  '''
  Bayesian Gaussian Linear Model
  '''

  tr['w'] = tpp.MultivariateNormal(a, sigma_w*t.eye(2))
  tr['obs'] = tpp.Normal(t.mm(tr['w'], data_x), sigma_y)



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(2,))


        self.log_s_mu = nn.Parameter(t.randn(2,2))



    def forward(self, tr):
        sigma_nn = t.mm(self.log_s_mu, self.log_s_mu.t())
        sigma_nn.add_(t.eye(2) * 1e-5)
        tr['w'] = tpp.MultivariateNormal(self.m_mu, sigma_nn)




model = tpp.Model(P, Q(), data_y)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

print("K=10")
for i in range(5000):
    opt.zero_grad()
    elbo = model.elbo(K=10)
    (-elbo).backward()
    opt.step()

    if 0 == i%500:
        print(elbo.item())


print("Approximate mu")
print(model.Q.m_mu)

print("Approximate Covariance")
print(model.Q.log_s_mu)
inferred_cov = t.mm(model.Q.log_s_mu, model.Q.log_s_mu.t())
inferred_cov.add_(t.eye(2)* 1e-5)
print(inferred_cov)

V_n = sigma_y * t.inverse(sigma_y * t.inverse(sigma_w*t.eye(2)) + t.mm(data_x,data_x.t()))
w_n = t.mm(V_n, t.mm(t.inverse(sigma_w*t.eye(2)),a.reshape(-1,1))) + (1/sigma_y) * t.mm(V_n, t.mm(data_x, data_y['obs'].reshape(-1,1)))


print("True mean weights")
print(w_n)

print("True weights covariance")
print(V_n)
