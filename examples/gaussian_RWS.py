import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from functorch.dim import dims

def P(nn.Module):
    def __init__(self):
        super().__init__()
        self.P_m_mu = nn.Parameter(t.zeros(5,))
        self.P_log_s_mu = nn.Parameter(t.zeros(5,))

        self.P_log_s_obs = nn.Parameter(t.zeros(5,))


    def forward(self, tr):
          tr['mu'] = tpp.Normal(self.P_m_mu, self.P_log_s_mu.exp())
          tr['obs'] = tpp.Normal(tr['mu'], self.P_log_s_obs.exp())



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.Q_m_mu = nn.Parameter(t.zeros(5,))

        self.Q_log_s_mu = nn.Parameter(t.zeros(5,))



    def forward(self, tr):
        tr['mu'] = tpp.Normal(self.m_mu, self.log_s_mu.exp())

# data = tpp.sample(P, "obs")
data = {'obs': t.tensor([ 0.9004, -3.7564,  0.4881, -1.1412,  0.2087])}

print(data)
model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(P.parameters(), lr=1E-3)


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
# K=1
# dims = tpp.make_dims(P, K)
# print("K={}".format(K))
# for i in range(10000):
#     opt.zero_grad()
#     elbo = model.elbo(dims=dims)
#     (-elbo).backward()
#     opt.step()
#
#     if 0 == i%1000:
#         print(elbo.item())
#
#
# print("Approximate mu")
# print(model.Q.m_mu)
#
# print("Approximate Covariance")
# print(model.Q.log_s_mu.exp())
#
# b_n = t.mm(t.inverse(t.eye(5) + t.eye(5)),tpp.dename(data['obs']).reshape(-1,1))
# A_n = t.inverse(t.eye(5) + t.eye(5))
#
# print("True mu")
# print(b_n)
#
# print("True covariance")
# print(t.diag(A_n))
