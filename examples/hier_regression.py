import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm


n_i = 100
N = 10
J = 10


weights = t.randn(N,n_i, names=('plate_1',None))
def P(tr):
  '''
  Bayesian Gaussian Linear Model
  '''

  tr['theta'] = tpp.Normal(t.zeros((10,)), 1)
  tr['z'] = tpp.Normal(tr['theta'], 1, sample_shape=N, sample_names='plate_1')

  tr['obs'] = tpp.Normal(tr['z'] @ weights, 1)



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_mu = nn.Parameter(t.zeros((J,)))
        self.log_theta_s = nn.Parameter(t.zeros((J,)))

        self.z_w = nn.Parameter(t.zeros((N,J), names = ('plate_1',None)))
        self.log_z_s = nn.Parameter(t.zeros((N,J), names = ('plate_1',None)))


    def forward(self, tr):
        tr['theta'] = tpp.Normal(self.theta_mu, self.log_theta_s.exp())
        tr['z'] = tpp.Normal(self.z_w*tr['theta'], self.log_z_s.exp())
        # print('Q')
        # print(tr['z'])
        # print(tr['z'].shape)



data_y = tpp.sample(P,"obs")
print(data_y['obs'].shape)
# d = tpp.sample(Q(),"z")

model = tpp.Model(P, Q(), data_y)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

print("K=5")
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(K=5)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())


thetas = []
zs = []
for i in range(10000):
    sample = tpp.sample(Q())
    thetas.append(sample['theta'].rename(None).flatten())
    zs.append(sample['z'].rename(None).flatten())

approx_theta_mean = t.mean(t.vstack(thetas).T, dim=1)
approx_theta_cov = t.round(t.cov(t.vstack(thetas).T), decimals=2)

# x is weights transposed
x = weights.rename(None)

#post_sigma_cov
inv = t.inverse(t.eye(100) + x.t() @ x)
post_sigma_cov = t.inverse(t.eye(10) + x @ inv @ x.t())

#post_sigma_mean
post_sigma_mean = post_sigma_cov @ (x @ inv @ data_y['obs'].mean(dim=0).rename(None).reshape(-1,1))

#post_z_cov
post_z_cov = t.inverse(t.eye(10) + x @ x.t())

#post_z_mean
post_z_mean = post_z_cov @ (x @ data_y['obs'].t() + post_sigma_mean)

print('Posterior theta mean')
print(post_sigma_mean)
print('Approximate Posterior theta mean')
print(approx_theta_mean)

print('Posterior theta cov')
print(post_sigma_cov)
print('Approximate Posterior theta cov')
print(approx_theta_cov)
