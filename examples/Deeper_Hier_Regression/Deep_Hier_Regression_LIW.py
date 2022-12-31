import torch as t
import torch.nn as nn
import alan

import argparse
import json
import numpy as np
import itertools
import time
import random

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)

seed_torch(0)

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

sizes = {'plate_muz2':2, 'plate_muz3':2, 'plate_muz4':2, 'plate_z':M, 'plate_obs':N}
if N == 30:
    d_z = 20
else:
    d_z = 5
x = {'x':t.load('weights_{0}_{1}.pt'.format(N,M)).rename('plate_muz2', 'plate_muz3', 'plate_muz4', 'plate_z', 'plate_obs', ...).to(device)}
# x = t.load('weights_{0}_{1}.pt'.format(N,M)).to(device)
def P(tr):
  '''
  Heirarchical Model
  '''

  tr.sample('mu_z1', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)), group='local')
  tr.sample('mu_z2', alan.Normal(tr['mu_z1'], t.ones(()).to(device)), plates='plate_muz2', group='local')
  tr.sample('mu_z3', alan.Normal(tr['mu_z2'], t.ones(()).to(device)), plates='plate_muz3', group='local')
  tr.sample('mu_z4', alan.Normal(tr['mu_z3'], t.ones(()).to(device)), plates='plate_muz4', group='local')
  tr.sample('psi_y', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)), group='local')
  tr.sample('psi_z', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)), group='local')

  tr.sample('z', alan.Normal(tr['mu_z4'] * t.ones((d_z)).to(device), tr['psi_z'].exp()), plates='plate_z')


  tr.sample('obs', alan.Normal((tr['z'] @ tr['x']), tr['psi_y'].exp()))



class Q(alan.QModule):
    def __init__(self):
        super().__init__()
        #mu_z1
        self.m_mu_z1 = nn.Parameter(t.zeros(()))
        self.log_theta_mu_z1 = nn.Parameter(t.zeros(()))
        #mu_z2
        self.m_mu_z2 = nn.Parameter(t.zeros((2,), names=('plate_muz2',)))
        self.log_theta_mu_z2 = nn.Parameter(t.zeros((2,), names=('plate_muz2',)))
        #mu_z3
        self.m_mu_z3 = nn.Parameter(t.zeros((2,2), names=('plate_muz2', 'plate_muz3')))
        self.log_theta_mu_z3 = nn.Parameter(t.zeros((2,2), names=('plate_muz2', 'plate_muz3')))
        #mu_z4
        self.m_mu_z4 = nn.Parameter(t.zeros((2,2,2), names=('plate_muz2', 'plate_muz3', 'plate_muz4')))
        self.log_theta_mu_z4 = nn.Parameter(t.zeros((2,2,2), names=('plate_muz2', 'plate_muz3', 'plate_muz4')))
        #psi_z
        self.m_psi_z = nn.Parameter(t.zeros(()))
        self.log_theta_psi_z = nn.Parameter(t.zeros(()))
        #psi_y
        self.m_psi_y = nn.Parameter(t.zeros(()))
        self.log_theta_psi_y = nn.Parameter(t.zeros(()))

        #z
        self.mu = nn.Parameter(t.zeros((2, 2, 2, M,d_z), names = ('plate_muz2', 'plate_muz3', 'plate_muz4', 'plate_z', None)))
        self.log_sigma = nn.Parameter(t.zeros((2, 2, 2, M,d_z), names = ('plate_muz2', 'plate_muz3', 'plate_muz4', 'plate_z', None)))


    def forward(self, tr):
        tr.sample('mu_z1', alan.Normal(self.m_mu_z1, self.log_theta_mu_z1.exp()))
        tr.sample('mu_z2', alan.Normal(self.m_mu_z2, self.log_theta_mu_z2.exp()))
        tr.sample('mu_z3', alan.Normal(self.m_mu_z3, self.log_theta_mu_z3.exp()))
        tr.sample('mu_z4', alan.Normal(self.m_mu_z4, self.log_theta_mu_z4.exp()))
        tr.sample('psi_z', alan.Normal(self.m_psi_z, self.log_theta_psi_z.exp()))
        tr.sample('psi_y', alan.Normal(self.m_psi_y, self.log_theta_psi_y.exp()))


        tr.sample('z', alan.Normal(self.mu, self.log_sigma.exp()))

data_y = {'obs':t.load('data_y_{0}_{1}.pt'.format(N, M)).rename('plate_muz2', 'plate_muz3', 'plate_muz4','plate_obs', 'plate_z').to(device)}
test_data_y = {'obs':t.load('test_data_y_{0}_{1}.pt'.format(N, M)).rename('plate_muz2', 'plate_muz3', 'plate_muz4', 'plate_obs', 'plate_z').to(device)}
for K in Ks:
    print(K,M,N)
    results_dict[N] = results_dict.get(N, {})
    results_dict[N][M] = results_dict[N].get(M, {})
    results_dict[N][M][K] = results_dict[N][M].get(K, {})
    elbos = []
    test_log_likelihoods = []
    times = []
    for i in range(5):
        per_seed_elbos = []
        start = time.time()
        seed_torch(i)

        model = alan.Model(P, Q(), data_y | x)
        model.to(device)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)
        scheduler = t.optim.lr_scheduler.StepLR(opt, step_size=10000, gamma=0.1)


        for j in range(50000):
            opt.zero_grad()
            elbo = model.elbo(K=K)
            (-elbo).backward()
            opt.step()
            scheduler.step()
            per_seed_elbos.append(elbo.item())
            if 0 == j%1000:
                print("Iteration: {0}, ELBO: {1:.2f}".format(j,elbo.item()))

        elbos.append(np.mean(per_seed_elbos[-50:]))
        times.append(time.time() - start)
        # test_log_likelihoods.append(model.test_log_like(dims=dim, test_data=test_data_y))
    results_dict[N][M][K] = {'lower_bound':np.mean(elbos),'std':np.std(elbos), 'elbos': elbos, 'avg_time':np.mean(times)}

file = 'results/results_LIW_N{0}_M{1}.json'.format(N,M)
with open(file, 'w') as f:
    json.dump(results_dict, f)
