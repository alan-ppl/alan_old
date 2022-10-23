import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from functorch.dim import dims
import argparse
import json
import numpy as np
import itertools

t.manual_seed(0)
parser = argparse.ArgumentParser(description='Run the Heirarchical regression task.')

parser.add_argument('N', type=int,
                    help='Scale of experiment')
parser.add_argument('M', type=int,
                    help='Number of groups')

args = parser.parse_args()

print('...', flush=True)


device = t.device("cuda" if t.cuda.is_available() else "cpu")

results_dict = {}

Ks = [1,5,10,15]
# Ns = [10,30]
# Ms = [10,50,100]

M = args.M
N = args.N

plate_muz2, plate_muz3, plate_muz4, plate_z, plate_obs = dims(5 , [2,2,2,M,N])
if N == 30:
    d_z = 20
else:
    d_z = 5
x = t.load('weights_{0}_{1}.pt'.format(N,M))[plate_muz2, plate_muz3, plate_muz4, plate_z, plate_obs].to(device)

def P(tr):
  '''
  Heirarchical Model
  '''

  tr['mu_z1'] = tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device))
  tr['mu_z2'] = tpp.Normal(tr['mu_z1'], t.ones(()).to(device), sample_dim=plate_muz2)
  tr['mu_z3'] = tpp.Normal(tr['mu_z2'], t.ones(()).to(device), sample_dim=plate_muz3)
  tr['mu_z4'] = tpp.Normal(tr['mu_z3'], t.ones(()).to(device), sample_dim=plate_muz4)
  tr['psi_z'] = tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device))
  tr['psi_y'] = tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device))

  tr['z'] = tpp.Normal(tr['mu_z4'] * t.ones((d_z)).to(device), tr['psi_z'].exp(), sample_dim=plate_z)

  tr['obs'] = tpp.Normal((tr['z'] @ x), tr['psi_y'].exp())



class Q(tpp.Q_module):
    def __init__(self):
        super().__init__()
        #mu_z1
        self.reg_param("m_mu_z1", t.zeros(()))
        self.reg_param("log_theta_mu_z1", t.zeros(()))
        #mu_z2
        self.reg_param("m_mu_z2", t.zeros((2,)), [plate_muz2])
        self.reg_param("log_theta_mu_z2", t.zeros((2,)), [plate_muz2])
        #mu_z3
        self.reg_param("m_mu_z3", t.zeros((2,2)), [plate_muz2, plate_muz3])
        self.reg_param("log_theta_mu_z3", t.zeros((2,2)), [plate_muz2, plate_muz3])
        #mu_z4
        self.reg_param("m_mu_z4", t.zeros((2,2,2)), [plate_muz2, plate_muz3, plate_muz4])
        self.reg_param("log_theta_mu_z4", t.zeros((2,2,2)), [plate_muz2, plate_muz3, plate_muz4])
        #psi_z
        self.reg_param("m_psi_z", t.zeros(()))
        self.reg_param("log_theta_psi_z", t.zeros(()))
        #psi_y
        self.reg_param("m_psi_y", t.zeros(()))
        self.reg_param("log_theta_psi_y", t.zeros(()))

        #z
        self.reg_param("mu", t.zeros((2, 2, 2, M,d_z)), [plate_muz2, plate_muz3, plate_muz4, plate_z])
        self.reg_param("log_sigma", t.zeros((2, 2, 2, M, d_z)), [plate_muz2, plate_muz3, plate_muz4, plate_z])


    def forward(self, tr):
        tr['mu_z1'] = tpp.Normal(self.m_mu_z1, self.log_theta_mu_z1.exp())
        tr['mu_z2'] = tpp.Normal(self.m_mu_z2, self.log_theta_mu_z2.exp())
        tr['mu_z3'] = tpp.Normal(self.m_mu_z3, self.log_theta_mu_z3.exp())
        tr['mu_z4'] = tpp.Normal(self.m_mu_z4, self.log_theta_mu_z4.exp())
        tr['psi_z'] = tpp.Normal(self.m_psi_z, self.log_theta_psi_z.exp())
        tr['psi_y'] = tpp.Normal(self.m_psi_y, self.log_theta_psi_y.exp())


        tr['z'] = tpp.Normal(self.mu, self.log_sigma.exp())
print(t.load('data_y_{0}_{1}.pt'.format(N, M)).shape)
data_y = {'obs':t.load('data_y_{0}_{1}.pt'.format(N, M))[plate_muz2, plate_muz3, plate_muz4, plate_obs, plate_z].to(device)}

for K in Ks:
    print(K,M,N)
    results_dict[N] = results_dict.get(N, {})
    results_dict[N][M] = results_dict[N].get(M, {})
    results_dict[N][M][K] = results_dict[N][M].get(K, {})
    elbos = []

    for i in range(5):

        t.manual_seed(i)

        model = tpp.Model(P, Q(), data_y)
        model.to(device)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)


        dim = tpp.make_dims(P, K)

        for i in range(50000):
            opt.zero_grad()
            elbo = model.elbo(dims=dim)
            (-elbo).backward()
            opt.step()

            if 0 == i%1000:
                print("Iteration: {0}, ELBO: {1:.2f}".format(i,elbo.item()))

        elbos.append(elbo.item())
    results_dict[N][M][K] = {'lower_bound':np.mean(elbos),'std':np.std(elbos), 'elbos': elbos}

file = 'results/results_N{0}_M{1}.json'.format(N,M)
with open(file, 'w') as f:
    json.dump(results_dict, f)