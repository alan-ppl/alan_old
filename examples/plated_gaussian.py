import torch as t
import torch.nn as nn
import tpp
from tpp.backend import vi
import tqdm
from functorch.dim import dims

N = 10
plate_1 = dims(1 , [N])
def P(tr):
    '''
    Bayesian Gaussian Model
    '''
    a = t.zeros(5,)
    tr['mu'] = tpp.MultivariateNormal(a, t.eye(5))
    # tr['obs'] = tpp.MultivariateNormal(tr['mu'], t.eye(5), sample_dim=plate_1)
    tr['obs'] = tpp.Normal(tr['mu'], t.tensor(1), sample_dim=plate_1)



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(5,))

        self.log_s_mu = nn.Parameter(t.zeros(5,))


    def forward(self, tr):
        tr['mu'] = tpp.MultivariateNormal(self.m_mu, t.diag(self.log_s_mu.exp()))

data = tpp.sample(P, "obs")
# data = {'obs': t.tensor([[[ 0.5851,  0.8783, -0.4380, -1.3839,  0.9538]],
#         [[ 0.3030,  0.8338, -2.2067, -1.8815,  3.3449]],
#         [[ 1.8357, -0.3146,  0.5771, -1.4885, -0.3881]],
#         [[-1.0334, -0.2395,  0.3544, -2.0973,  1.8586]],
#         [[ 0.5752, -0.9763, -1.0950, -0.2201,  0.4888]]])[plate_1, :]}
print(data)
model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K = 3
Ks = tpp.make_dims(P, K)
print("K={}".format(K))
for i in range(15000):
    opt.zero_grad()
    elbo = model.elbo(dims=Ks)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())

print("Approximate mu")
print(model.Q.m_mu)

print("Approximate Covariance")
print(model.Q.log_s_mu.exp())


data_obs = tpp.dename(data['obs'])

b_n = t.mm(t.inverse(t.eye(5) + 1/N * t.eye(5)),data_obs.mean(axis=0).reshape(-1,1))
A_n = t.inverse(t.eye(5) + 1/N * t.eye(5)) * 1/N

print("True mu")
print(b_n)

print("True covariance")
print(t.diag(A_n))

print((t.abs(b_n - model.Q.m_mu.reshape(-1,1))<0.1).all())
print((t.abs(A_n - t.diag(model.Q.log_s_mu.exp()))<0.1).all())
