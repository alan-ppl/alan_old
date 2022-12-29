import torch as t
import torch.nn as nn
import alan
import tqdm
from functorch.dim import dims

'''
Test posterior inference with a Gaussian with plated observations
'''
N = 10
plate_1 = dims(1 , [N])
a = t.zeros(5,)

gen_probs = t.tensor([0.1,0.1,0.7,0.05,0.05])
class P(nn.Module):
    def __init__(self):
        super().__init__()
        self.prob_mu = nn.Parameter(t.tensor([0.1,0.8,0.05,0.05,0.05]))
        self.prob_sigma = nn.Parameter(t.tensor([0.1,0.8,0.05,0.05,0.05]))
        #self.prob = gen_probs


    def forward(self, tr):
        tr['mu'] = alan.Categorical(logits = self.prob_mu, sample_K=False)
        tr['sigma'] = alan.Categorical(logits = self.prob_sigma, sample_K=False)
        tr['obs'] = alan.Normal(t.ones(5,)*tr['mu'], t.ones(5,)*tr['sigma'].exp(), sample_dim=plate_1)

class Q(alan.Q_module):
    def __init__(self):
        super().__init__()
        self.reg_param('prob_sigma', t.randn(5))
        self.reg_param('prob_mu', t.randn(5))


    def forward(self, tr):
        tr['mu'] = alan.Categorical(logits = self.prob_mu, sample_K=False)
        tr['sigma'] = alan.Categorical(logits = self.prob_sigma, sample_K=False)

#data = alan.sample(P(), 'obs')

data ={'obs': t.tensor([[ 1.6145,  0.0647,  0.5294,  1.7702,  3.6971],
        [ 2.3284,  2.7149,  2.2479,  1.5902,  2.5105],
        [ 2.2991,  0.5888,  2.3346,  2.6348,  3.4941],
        [ 3.0723,  2.9116,  2.8320,  2.7947,  2.0869],
        [ 3.0348,  1.3412,  3.3689,  5.1772,  0.8872],
        [ 1.0979,  2.1693,  1.1301,  1.8357,  1.6108],
        [ 2.4647, -0.0914,  1.6831,  3.1719,  3.4021],
        [ 1.5611,  2.6564,  2.5020,  1.8012,  3.4010],
        [ 0.9619,  2.5353,  3.1481,  3.0188,  3.0644],
        [ 2.2749,  1.3944,  2.9946,  2.8589,  0.6718]])[plate_1]}



model = alan.Model(P(), Q(), data)


opt = t.optim.Adam(model.parameters(), lr=1E-3)

K = 5
dim = alan.make_dims(P(), K, exclude=['mu', 'sigma'])

for i in range(5):
    opt.zero_grad()
    theta_loss, phi_loss = model.rws(dims=dim)
    (theta_loss + phi_loss).backward()
    # elbo = model.elbo(dims=dim)
    # (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(phi_loss.item())

        # print(elbo.item())

print(t.softmax(model.P.prob_sigma, 0))
print(t.softmax(model.P.prob_mu, 0))
print(gen_probs)
print(t.softmax(model.Q.prob_sigma, 0))
print(t.softmax(model.Q.prob_mu, 0))
print()
