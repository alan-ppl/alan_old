import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from functorch.dim import dims
import matplotlib.pyplot as plt

theta_size = 10

N = 10
n_i = 100
plate_1, plate_2 = dims(2 , [N,n_i])
x = t.randn(N,n_i,theta_size)[plate_1,plate_2,:]

j,k = dims(2)
def P(tr):
  '''
  Heirarchical Model
  '''

  tr['theta'] = tpp.Normal(t.zeros((theta_size,)), 1)
  tr['z'] = tpp.Normal(tr['theta'], 1, sample_dim=plate_1)
  tr['obs'] = tpp.Normal((x.t() @ tr['z']), 1)


class Q(tpp.Q_module):
    def __init__(self):
        super().__init__()
        self.reg_param("theta_mu", t.zeros((theta_size,)))
        self.reg_param("log_theta_s", t.zeros((theta_size,)))

        self.reg_param("z_w", t.zeros((N,theta_size)), [plate_1])
        self.reg_param("z_b", t.zeros((N,theta_size)), [plate_1])
        self.reg_param("log_z_s", t.randn((N,theta_size,)), [plate_1])


    def forward(self, tr):


        tr['theta'] = tpp.Normal(self.theta_mu, self.log_z_s.exp())
        tr['z'] = tpp.Normal(tr['theta']@self.z_w + self.z_b, self.log_z_s.exp())





data_y = tpp.sample(P,"obs")

model = tpp.Model(P, Q(), data_y)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=5
dim = tpp.make_dims(P, K, [plate_1])
print("K={}".format(K))

iters = 20
elbos = []
for i in range(iters):
    opt.zero_grad()
    elbo = model.elbo(dims=dim)
    (-elbo).backward()
    opt.step()
    elbos.append(elbo.item())
    if 0 == i%1000:
        print("Iteration: {0}, ELBO: {1:.2f}".format(i,elbo.item()))

# zs = []
# for i in range(1000):
#     sample = tpp.sample(Q())
#     zs.append(tpp.dename(sample['z']).flatten())
#
# approx_theta_mean = t.mean(t.vstack(thetas).T, dim=1)
# approx_theta_cov = t.cov(t.vstack(thetas).T)
#
# approx_z_mean = t.mean(t.vstack(zs).T, dim=1)
# approx_z_cov = t.cov(t.vstack(zs).T)


#Theta posterior
x_sum = sum([tpp.dename(x)[i].t() @ t.inverse(t.eye(n_i) + tpp.dename(x)[i] @ tpp.dename(x)[i].t()) @ tpp.dename(x)[i] for i in range(N)])
y_sum = sum([tpp.dename(x)[i].t() @ t.inverse(t.eye(n_i) + tpp.dename(x)[i] @ tpp.dename(x)[i].t()) @ tpp.dename(data_y['obs'])[i] for i in range(N)])

post_theta_cov = t.eye(theta_size) + x_sum
post_theta_mean = t.inverse(post_theta_cov) @ y_sum

# print('Posterior theta mean')
# print(post_theta_mean)
# print('Approximate Posterior theta mean')
# print(model.Q.theta_mu)
#
# print('Posterior theta cov')
# print(t.round(t.inverse(post_theta_cov),decimals=2))
# print('Approximate Posterior theta cov')
# print(model.Q.log_theta_s.exp())


# print('Posterior z mean')
# print(t.round(post_z_mean, decimals=2).shape)
# print('Approximate Posterior z mean')
# print(t.round(approx_z_mean.reshape(10,-1), decimals=2).shape)
#
# print('Posterior z cov')
# print(t.round(post_z_cov, decimals=2).shape)
# print('Approximate Posterior z cov')
# print(t.round(approx_z_cov, decimals=2).shape)


### plotting elbos
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(range(iters),elbos, label="K=5")
ax.legend()
plt.savefig('K5.pdf')
