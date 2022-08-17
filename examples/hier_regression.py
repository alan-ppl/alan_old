import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from torchdim import dims

n_i = 100
J = 10

N = 10
plate_1 = dims(1 , [N])
weights = t.randn(n_i,N)[:,plate_1]

j,k = dims(2)
def P(tr):
  '''
  Bayesian Gaussian Linear Model
  '''

  tr['theta'] = tpp.Normal(t.zeros((J,)), 1)
  tr['z'] = tpp.Normal(tr['theta'], 1, sample_dim=plate_1)
  tr['obs'] = tpp.Normal((weights * tr['z'][k]).sum(k), 1)


class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_mu = nn.Parameter(t.zeros((J,)))
        self.log_theta_s = nn.Parameter(t.zeros((J,)))

        self.z_w = nn.Parameter(t.zeros((N,J)))
        self.z_b = nn.Parameter(t.zeros((N,J)))
        self.log_z_s = nn.Parameter(t.zeros((N,J)))


    def forward(self, tr):
        z_w = self.z_w[plate_1]
        z_b = self.z_b[plate_1]
        log_z_s = self.log_z_s[plate_1]

        tr['theta'] = tpp.Normal(self.theta_mu, self.log_theta_s.exp())
        tr['z'] = tpp.Normal(z_w*tr['theta'] + z_b, log_z_s.exp())



data_y = tpp.sample(P,"obs")

model = tpp.Model(P, Q(), data_y)

opt = t.optim.Adam(model.parameters(), lr=1E-2)

K=25
dim = tpp.make_dims(P, K, [plate_1])
print("K={}".format(K))

for i in range(20000):
    opt.zero_grad()
    elbo = model.elbo(dims=dim)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())


thetas = []
zs = []
for i in range(10):
    sample = tpp.sample(Q())
    thetas.append(tpp.dename(sample['theta']).flatten())
    zs.append(tpp.dename(sample['z']).flatten())

approx_theta_mean = t.mean(t.vstack(thetas).T, dim=1)
approx_theta_cov = t.cov(t.vstack(thetas).T)

approx_z_mean = t.mean(t.vstack(zs).T, dim=1)
approx_z_cov = t.cov(t.vstack(zs).T)

# x is weights transposed
x = tpp.dename(weights).t()

#post_theta_cov
inv = t.inverse(t.eye(100) + x @ x.t())
post_theta_cov = t.inverse(t.eye(10) + x.t() @ inv @ x)

#post_theta_mean
post_theta_mean = post_theta_cov @ (x.t() @ inv @ tpp.dename(data_y['obs']).mean(axis=0))

#post_z_cov
post_z_cov = t.inverse(t.eye(100) + x @ x.t())

#post_z_mean
post_z_mean = post_z_cov @ (x.t() @ data_y['obs'] + post_theta_mean)

print('Posterior theta mean')
print(t.round(post_theta_mean, decimals=2).shape)
print('Approximate Posterior theta mean')
print(t.round(approx_theta_mean, decimals=2).shape)

print('Posterior theta cov')
print(t.round(post_theta_cov, decimals=2).shape)
print('Approximate Posterior theta cov')
print(t.round(approx_theta_cov, decimals=2).shape)


print('Posterior z mean')
print(t.round(post_z_mean, decimals=2).shape)
print('Approximate Posterior z mean')
print(t.round(approx_z_mean.reshape(10,-1), decimals=2).shape)

print('Posterior z cov')
print(t.round(post_z_cov, decimals=2).shape)
print('Approximate Posterior z cov')
print(t.round(approx_z_cov, decimals=2).shape)
