import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from torchdim import dims
from tpp.dims import *


plate_1 = dims(1)
plate_1.size = 5
def P(tr):
    '''
    Bayesian Gaussian Model
    '''
    a = t.zeros(5,)
    tr['mu'] = tpp.MultivariateNormal(a, t.eye(5))
    print('tr_mu')
    print(tr['mu'])
    tr['obs'] = tpp.MultivariateNormal(tr['mu'], t.eye(5), sample_dim=plate_1)
    print('tr_obs')
    print(tr['obs'])



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(5,))

        self.log_s_mu = nn.Parameter(t.zeros(5,))


    def forward(self, tr):
        tr['mu'] = tpp.MultivariateNormal(self.m_mu, t.diag(self.log_s_mu.exp()))

data = tpp.sample(P, "obs")
# data = {'obs': t.tensor([ 0.9004, -3.7564,  0.4881, -1.1412,  0.2087])}
K_obs = dims(1)
K_obs.size = 1
# data['obs'] = data['obs'].unsqueeze(0)[K_obs,:]
obs = t.tensor([[[ 0.5319,  0.4274, -0.7725, -1.1542, -0.7040]],

        [[-0.1281, -0.7801, -0.5250, -0.8313,  1.6599]],

        [[ 1.0722,  1.2969, -0.7785, -0.9243, -0.4684]],

        [[ 1.0723, -0.1205,  0.1732, -2.0058,  0.7741]],

        [[ 3.2124,  0.1402, -0.9065, -2.8523,  0.1510]]])
print(obs.reshape(1,5,5))
print(obs.shape)
data = {'obs': t.tensor([[[ 0.5319,  0.4274, -0.7725, -1.1542, -0.7040],
         [-0.1281, -0.7801, -0.5250, -0.8313,  1.6599],
         [ 1.0722,  1.2969, -0.7785, -0.9243, -0.4684],
         [ 1.0723, -0.1205,  0.1732, -2.0058,  0.7741],
         [ 3.2124,  0.1402, -0.9065, -2.8523,  0.1510]]])[K_obs, plate_1, :]}

print(data)
model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)


K, K_mu = dims(2)
K.size = 20
K_mu.size = 20
make_K([K,K_mu])
make_plate(plate_1)
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
