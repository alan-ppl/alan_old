import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from functorch.dim import dims
import numpy as np

def P(tr):
  '''
  Bayesian Gaussian Model
  '''
  a = t.zeros(5,)
  tr['mu'] = tpp.Normal(a, t.ones(5,))
  tr['obs'] = tpp.Normal(tr['mu'], t.ones(5,))



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(5,))

        self.log_s_mu = nn.Parameter(t.zeros(5,))



    def forward(self, tr):
        tr['mu'] = tpp.Normal(self.m_mu, self.log_s_mu.exp())


data = tpp.sample(P, "obs")
test_data = tpp.sample(P, "obs")





model = tpp.Model(P, Q(), data)


opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=5
dim = tpp.make_dims(P, K)
print("K={}".format(K))
for i in range(5000):
    opt.zero_grad()
    theta_loss, phi_loss = model.rws(dims=dim)
    (theta_loss + phi_loss).backward()
    opt.step()

    if 0 == i%1000:
        print(phi_loss.item())
        # print(theta_loss.item())


print("Approximate mu")
print(model.Q.m_mu)

print("Approximate Covariance")
print(model.Q.log_s_mu.exp()**2)

b_n = t.mm(t.inverse(t.eye(5) + t.eye(5)),tpp.dename(data['obs']).reshape(-1,1))
A_n = t.inverse(t.eye(5) + t.eye(5))

print("True mu")
print(b_n)

print("True covariance")
print(t.diag(A_n))
