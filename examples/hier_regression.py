import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm


n_i = 100
J = 10

N = 10
plate_1 = dims(1 , [N])
weights = t.randn(N,n_i)[plate_1]
def P(tr):
  '''
  Bayesian Gaussian Linear Model
  '''

  tr['theta'] = tpp.Normal(t.zeros((J,)), 1)
  tr['z'] = tpp.Normal(tr['theta'], 1, , sample_dim=plate_1)

  tr['obs'] = tpp.Normal(tr['z'] @ weights, 1)



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_mu = nn.Parameter(t.zeros((J,)))
        self.log_theta_s = nn.Parameter(t.zeros((J,)))

        self.z_w = nn.Parameter(t.zeros((N,J)))
        self.z_b = nn.Parameter(t.zeros((N,J)))
        self.log_z_s = nn.Parameter(t.zeros((N,J)))


    def forward(self, tr):
        w_c = self.z_w[plate_1]
        b_c = self.z_b[plate_1]
        log_s_c = self.log_z_s[plate_1]


        tr['theta'] = tpp.Normal(self.theta_mu, self.log_theta_s.exp())
        tr['z'] = tpp.Normal(z_w*tr['theta'] + z_b, log_z_s.exp())
        # print('Q')
        # print(tr['z'])
        # print(tr['z'].shape)



data_y = tpp.sample(P,"obs")
print(data_y['obs'].shape)
# d = tpp.sample(Q(),"z")

model = tpp.Model(P, Q(), data_y)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=2
dims = tpp.make_dims(P, K, [plate_1])
print("K={}".format(K))

for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(dims=dims)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())


thetas = []
zs = []
for i in range(10000):
    sample = tpp.sample(Q())
    thetas.append(tpp.dename(sample['theta']).rename(None).flatten())
    zs.append(tpp.dename(sample['z']).rename(None).flatten())

approx_theta_mean = t.mean(t.vstack(thetas).T, dim=1)
approx_theta_cov = t.cov(t.vstack(thetas).T)

# x is weights transposed
x = weights.rename(None)

#post_sigma_cov
inv = t.inverse(t.eye(100) + x.t() @ x)
post_sigma_cov = t.inverse(t.eye(10) + x @ inv @ x.t())

#post_sigma_mean
post_sigma_mean = post_sigma_cov @ (x @ inv @ tpp.dename(data_y['obs']).mean(dim=0).rename(None).reshape(-1,1))

#post_z_cov
post_z_cov = t.inverse(t.eye(10) + x @ x.t())

#post_z_mean
post_z_mean = post_z_cov @ (x @ tpp.dename(data_y['obs']).t() + post_sigma_mean)

print('Posterior theta mean')
print(t.round(post_sigma_mean, decimals=2))
print('Approximate Posterior theta mean')
print(t.round(approx_theta_mean, decimals=2))

print('Posterior theta cov')
print(t.round(post_sigma_cov, decimals=2))
print('Approximate Posterior theta cov')
print(t.round(approx_theta_cov, decimals=2))
