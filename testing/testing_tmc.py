import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from functorch.dim import dims

theta_size = 10

N =10
n_i = 100
plate_1, plate_2 = dims(2 , [N,n_i])
x = t.randn(N,n_i,theta_size)[plate_1,plate_2,:].to(device)

j,k = dims(2)

theta_mean = t.zeros(theta_size).to(device)
theta_sigma = t.ones(theta_size).to(device)

z_sigma = t.ones(theta_size).to(device)

obs_sigma = t.tensor(1).to(device)
def P(tr):
  '''
  Heirarchical Model
  '''
  tr['theta'] = tpp.MultivariateNormal(theta_mean, t.diag(theta_sigma))
  tr['z'] = tpp.MultivariateNormal(tr['theta'], t.diag(z_sigma), sample_dim=plate_1)

  tr['obs'] = tpp.Normal((x @ tr['z']), obs_sigma)


class Q(tpp.Q_module):
    def __init__(self):
        super().__init__()
        self.reg_param("theta_mu", t.zeros((theta_size,)))
        self.reg_param("theta_s", t.randn((theta_size,theta_size)))

        self.reg_param("mu", t.zeros((N,theta_size)), [plate_1])
        self.reg_param("A", t.zeros((N,theta_size)), [plate_1])
        self.reg_param("z_s", t.randn((N,theta_size,theta_size)), [plate_1])


    def forward(self, tr):
        sigma_theta = self.theta_s @ self.theta_s.mT
        eye = t.eye(theta_size).to(device)
        sigma_theta = sigma_theta + eye * 0.001

        sigma_z = self.z_s @ self.z_s.mT
        z_eye = eye * 0.001
        sigma_z = sigma_z + z_eye

        tr['theta'] = tpp.MultivariateNormal(self.theta_mu, sigma_theta)
        tr['z'] = tpp.MultivariateNormal(tr['theta']@self.A + self.mu, sigma_z)

data = tpp.sample(P, "obs")
test_data = tpp.sample(P, "obs")

print(data)
model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=5
dims = tpp.make_dims(P, K)
print("K={}".format(K))
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(dims=dims)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())


print("Approximate mu")
print(model.Q.m_mu)

print("Approximate Covariance")
print(model.Q.log_s_mu.exp()**2)

b_n = t.mm(t.inverse(t.eye(5) + t.eye(5)),tpp.dename(data['obs']).reshape(-1,1))
A_n = t.inverse(t.eye(5) + t.eye(5))

print("True mu")
print(b_n)

print("True covariance")
print(t.diag(A_n))
