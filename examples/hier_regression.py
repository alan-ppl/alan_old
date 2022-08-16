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
weights = t.randn(N,n_i)[plate_1]
def P(tr):
  '''
  Bayesian Gaussian Linear Model
  '''

  tr['theta'] = tpp.Normal(t.zeros((J,)), 1)
  tr['z'] = tpp.Normal(tr['theta'], 1, sample_dim=plate_1)
  # print(tr['z'].shape)
  # print(weights.shape)
  i,j = dims(2)
  # print(weights.T)
  # print((weights[j,plate_1] * tr['z'][i]).sum(i))

  tr['obs'] = tpp.Normal((weights * tr['z'][i]).sum(i).order(plate_1), 1)



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_mu = nn.Parameter(t.zeros((J,)))
        self.log_theta_s = nn.Parameter(t.zeros((J,)))

        self.z_w = nn.Parameter(t.zeros((N,J)))[plate_1]
        self.z_b = nn.Parameter(t.zeros((N,J)))[plate_1]
        self.log_z_s = nn.Parameter(t.zeros((N,J)))[plate_1]


    def forward(self, tr):

        tr['theta'] = tpp.Normal(self.theta_mu, self.log_theta_s.exp())
        tr['z'] = tpp.Normal(self.z_w*tr['theta'] + self.z_b, self.log_z_s.exp())
        # print('Q')
        # print(tr['z'])
        # print(tr['z'].shape)



data_y = tpp.sample(P,"obs")
# d = tpp.sample(Q(),"z")

model = tpp.Model(P, Q(), data_y)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=2
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
for i in range(10000):
    sample = tpp.sample(Q())
    thetas.append(tpp.dename(sample['theta']).flatten())
    zs.append(tpp.dename(sample['z']).flatten())

approx_theta_mean = t.mean(t.vstack(thetas).T, dim=1)
approx_theta_cov = t.cov(t.vstack(thetas).T)

approx_z_mean = t.mean(t.vstack(zs).T, dim=1)
approx_z_cov = t.cov(t.vstack(zs).T)

# x is weights transposed
x = tpp.dename(weights)

#post_theta_cov
inv = t.inverse(t.eye(100) + x.t() @ x)
post_theta_cov = t.inverse(t.eye(10) + x @ inv @ x.t())

#post_theta_mean
post_theta_mean = post_theta_cov @ (x @ inv @ tpp.dename(data_y['obs']).t())

#post_z_cov
post_z_cov = t.inverse(t.eye(10) + x @ x.t())

#post_z_mean
post_z_mean = post_z_cov @ (x @ tpp.dename(data_y['obs']).t() + post_theta_mean)

print('Posterior theta mean')
print(t.round(post_theta_mean, decimals=2))
print('Approximate Posterior theta mean')
print(t.round(approx_theta_mean, decimals=2))

print('Posterior theta cov')
print(t.round(post_theta_cov, decimals=2))
print('Approximate Posterior theta cov')
print(t.round(approx_theta_cov, decimals=2))


print('Posterior z mean')
print(t.round(post_z_mean, decimals=2))
print('Approximate Posterior z mean')
print(t.round(approx_z_mean, decimals=2))

print('Posterior z cov')
print(t.round(post_z_cov, decimals=2))
print('Approximate Posterior z cov')
print(t.round(approx_z_cov, decimals=2))
