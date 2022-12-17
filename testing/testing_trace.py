import torch as t
import torch.nn as nn
from tpp.torchdim_dist import *
from tpp.traces import Trace, TraceLogP, TraceSampleLogQ, Q, sample, Model
import tqdm
from functorch.dim import dims
from tpp.utils import *


plate_1 = dims(1 , [5])
def P(tr):
    '''
    Bayesian Gaussian Model
    '''
    a = t.zeros(5,)
    tr.sample("mu", Normal(a, t.ones(5,)), plate=plate_1)
    tr.sample("obs", Normal(tr['mu'], t.ones(5,)), plate=plate_1)



class Q_(Q):
    def __init__(self):
        super().__init__()
        self.reg_param('m_mu', t.zeroes(5,5).rename('plate_1', None))

    def forward(self, tr):
        tr.sample("mu", Normal(self.m_mu, 1))


data = sample(P, "obs")
# data = {'obs': t.tensor(t.tensor([[[ 0.5851,  0.8783, -0.4380, -1.3839,  0.9538]],
#         [[ 0.3030,  0.8338, -2.2067, -1.8815,  3.3449]],
#         [[ 1.8357, -0.3146,  0.5771, -1.4885, -0.3881]],
#         [[-1.0334, -0.2395,  0.3544, -2.0973,  1.8586]],
#         [[ 0.5752, -0.9763, -1.0950, -0.2201,  0.4888]]])[plate_1, :]}

print(data)
model = Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)


K = 20


print("K={}".format(K))
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(K=K)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())
