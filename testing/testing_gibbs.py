import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from functorch.dim import dims
from tpp.utils import *


plate_1 = dims(1 , [5])
def P(tr):
    '''
    Bayesian Gaussian Model
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

data = tpp.sample(P, "obs")
# data = {'obs': t.tensor(t.tensor([[[ 0.5851,  0.8783, -0.4380, -1.3839,  0.9538]],
#         [[ 0.3030,  0.8338, -2.2067, -1.8815,  3.3449]],
#         [[ 1.8357, -0.3146,  0.5771, -1.4885, -0.3881]],
#         [[-1.0334, -0.2395,  0.3544, -2.0973,  1.8586]],
#         [[ 0.5752, -0.9763, -1.0950, -0.2201,  0.4888]]])[plate_1, :]}

print(data)
model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)


K = 20

dim = tpp.make_dims(P, K)

print("K={}".format(K))
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(dims=dim)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())

print(model.importance_sample(dims=dim))
print(model.importance_sample(dims=dim))
print(model.importance_sample(dims=dim))
print(model.importance_sample(dims=dim))
