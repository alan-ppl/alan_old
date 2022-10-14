import torch as t
import torch.nn as nn
import tpp
from tpp.backend import vi
import tqdm
from functorch.dim import dims

'''
Test posterior inference with a Gaussian with plated observations
'''
N = 10
plate_1 = dims(1 , [N])
a = t.zeros(5,)
# def P(tr):
#   '''
#   Bayesian Heirarchical Gaussian Model
#   '''
#   tr['mu'] = tpp.Bernoulli(t.tensor(0.3))
#   tr['obs'] = tpp.MultivariateNormal(t.ones(5,)*tr['mu'], t.diag(t.ones(5,)), sample_dim=plate_1)

class P(nn.Module):
    def __init__(self):
        super().__init__()
        self.prob = nn.Parameter(t.tensor([0.1,0.8,0.05,0.05,0.05]))
        # self.prob = t.tensor([0.1,0.8,0.05,0.05,0.05])
        print(t.softmax(self.prob, 0))

    def forward(self, tr):
        tr['mu'] = tpp.Categorical(logits = self.prob)
        tr['obs'] = tpp.MultivariateNormal(t.ones(5,)*tr['mu'], t.diag(t.ones(5,)), sample_dim=plate_1)

class Q(tpp.Q_module):
    def __init__(self):
        super().__init__()
        self.reg_param('prob', t.randn(5))



    def forward(self, tr):
        tr['mu'] = tpp.Categorical(logits = self.prob)

data = tpp.sample(P(), 'obs')

data ={'obs': t.tensor([[1.9503, 3.1042, 0.5958, 3.1034, 3.2131],
        [3.5996, 0.4358, 3.2042, 2.0266, 1.2166],
        [1.6207, 2.5711, 3.7055, 2.4605, 2.1354],
        [1.6843, 1.5303, 2.7101, 2.2667, 1.5363],
        [0.1846, 0.8784, 1.6729, 1.6051, 0.6131],
        [2.8810, 1.4781, 2.7638, 1.0168, 1.7689],
        [0.4804, 1.6845, 1.9839, 1.0556, 0.8191],
        [2.2323, 1.0172, 1.6986, 2.1626, 2.0412],
        [2.6557, 2.9926, 0.6813, 1.4802, 1.2957],
        [2.0657, 1.6080, 0.6045, 1.0571, 3.1758]])[plate_1]}



model = tpp.Model(P(), Q(), data)


opt = t.optim.Adam(model.parameters(), lr=1E-3)

K = 5
dim = tpp.make_dims(P(), K)

for i in range(15000):
    opt.zero_grad()
    theta_loss, phi_loss = model.rws(dims=dim)
    (theta_loss + phi_loss).backward()
    # elbo = model.elbo(dims=dim)
    # (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(phi_loss.item())

        # print(elbo.item())

print(t.softmax(model.P.prob, 0))
print(t.softmax(model.Q.prob, 0))
