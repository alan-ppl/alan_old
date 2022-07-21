import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from torchdim import dims


plate_1 = dims(1)
plate_1.size = 5
def P(tr):
    '''
    Bayesian Gaussian Model
    '''
    a = t.zeros(5,)
    tr['mu'] = tpp.MultivariateNormal(a, t.eye(5), sample_dim=plate_1)

    tr['obs'] = tpp.MultivariateNormal(tr['mu'], t.eye(5))



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(5,5))

        self.log_s_mu = nn.Parameter(t.zeros(5,))


    def forward(self, tr):
        tr['mu'] = tpp.MultivariateNormal(self.m_mu, t.diag(self.log_s_mu.exp()), sample_dim=plate_1)

data = tpp.sample(P, "obs")
# data = {'obs': t.tensor([ 0.9004, -3.7564,  0.4881, -1.1412,  0.2087])}

print(data)
model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)


K, K_mu = dims(2)
K.size = 5
K_mu.size = 5
dims = {'K':K, 'K_mu':K_mu, 'plate_1':plate_1}
print("K={}".format(K.size))
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(dims=dims)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())

print(model.importance_sample(K=5))
