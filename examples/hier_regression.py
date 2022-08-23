import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from functorch.dim import dims


theta_size = 10

M = 10
N_i = 10
plate_1, plate_2 = dims(2 , [M,N_i])
x = t.randn(M,N_i)[plate_1,plate_2]

print(plate_1)
def P(tr):
  '''
  Heirarchical Model
  '''

  tr['mu_z'] = tpp.Normal(t.zeros(()), 1)
  tr['psi_z'] = tpp.Normal(t.zeros(()), 1)
  tr['z'] = tpp.Normal(tr['mu_z'], tr['psi_z'].exp(), sample_dim=plate_1)
  tr['psi_y'] = tpp.Normal(t.zeros(()), 1)
  tr['obs'] = tpp.Normal((x.t() * tr['z']), tr['psi_y'].exp())


class Q(tpp.Q_module):
    def __init__(self):
        super().__init__()
        #mu_z
        self.reg_param("m_mu_z", t.zeros(()))
        self.reg_param("log_theta_mu_z", t.zeros(()))
        #psi_z
        self.reg_param("m_psi_z", t.zeros(()))
        self.reg_param("log_theta_psi_z", t.zeros(()))
        #psi_y
        self.reg_param("m_psi_y", t.zeros(()))
        self.reg_param("log_theta_psi_y", t.zeros(()))

        #z
        self.reg_param("z_w", t.zeros((M,)), [plate_1])
        self.reg_param("z_b", t.zeros((M,)), [plate_1])
        self.reg_param("log_z_w", t.randn((M,)), [plate_1])
        self.reg_param("log_z_b", t.randn((M,)), [plate_1])


    def forward(self, tr):
        tr['mu_z'] = tpp.Normal(self.m_mu_z, self.log_theta_mu_z.exp())
        tr['psi_z'] = tpp.Normal(self.m_psi_z, self.log_theta_psi_z.exp())
        tr['psi_y'] = tpp.Normal(self.m_psi_y, self.log_theta_psi_y.exp())


        tr['z'] = tpp.Normal(tr['mu_z']*self.z_w + self.z_b, (self.log_z_w * tr['psi_z'] + self.log_z_b).exp())





data_y = tpp.sample(P,"obs")

model = tpp.Model(P, Q(), data_y)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=1
dim = tpp.make_dims(P, K, [plate_1])
print("K={}".format(K))

for i in range(50000):
    opt.zero_grad()
    elbo = model.elbo(dims=dim)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print("Iteration: {0}, ELBO: {1:.2f}".format(i,elbo.item()))
