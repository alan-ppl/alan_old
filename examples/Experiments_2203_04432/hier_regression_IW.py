import torch as t
import torch.nn as nn
import tpp

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


sizes = {'plate_1':M, 'plate_2':N}
if N == 30:
    d_z = 20
else:
    d_z = 5
x = {'x':t.load('weights_{0}_{1}.pt'.format(N,M)).rename('plate_1','plate_2',...).to(device)}

def P(tr):
  '''
  Heirarchical Model
  '''

  tr.sample('mu_z', tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device)))
  tr.sample('psi_z', tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device)))
  tr.sample('psi_y', tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device)))

  tr.sample('z', tpp.Normal(tr['mu_z'] * t.ones((d_z)).to(device), tr['psi_z'].exp()), plate='plate_1')

  tr.sample('obs', tpp.Normal((tr['z'] @ tr['x']), tr['psi_y'].exp()))




class Q(tpp.Q):
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
        self.reg_param("mu", t.zeros((M,d_z)), ['plate_1'])
        self.reg_param("log_sigma", t.zeros((M, d_z)), ['plate_1'])


    def forward(self, tr):
        tr.sample('mu_z', tpp.Normal(self.m_mu_z, self.log_theta_mu_z.exp()), multi_samples=False)
        tr.sample('psi_z', tpp.Normal(self.m_psi_z, self.log_theta_psi_z.exp()), multi_samples=False)
        tr.sample('psi_y', tpp.Normal(self.m_psi_y, self.log_theta_psi_y.exp()), multi_samples=False)


        tr.sample('z', tpp.Normal(self.mu, self.log_sigma.exp()))

data_y = {'obs':t.load('data_y_{0}_{1}.pt'.format(N, M)).rename('plate_1','plate_2').to(device)}

for K in Ks:
    print(K,M,N)
    results_dict[N] = results_dict.get(N, {})
    results_dict[N][M] = results_dict[N].get(M, {})
    results_dict[N][M][K] = results_dict[N][M].get(K, {})
    elbos = []
    times = []
    lrs = []
    for i in range(5):
        per_seed_elbos = []
        seed_torch(i)
        start = time.time()
        lr = []
        t.manual_seed(i)

        model = tpp.Model(P, Q(), data_y | x)
        model.to(device)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)
        scheduler = t.optim.lr_scheduler.StepLR(opt, step_size=10000, gamma=0.1)

        for i in range(50000):
            opt.zero_grad()
            elbo = model.elbo(K=K)
            (-elbo).backward()
            opt.step()
            scheduler.step()
            per_seed_elbos.append(elbo.item())
            if 0 == i%1000:
                print("Iteration: {0}, ELBO: {1:.2f}".format(i,elbo.item()))

        elbos.append(np.mean(per_seed_elbos[-50:]))
        times.append(time.time() - start)
    results_dict[N][M][K] = {'lower_bound':np.mean(elbos),'std':np.std(elbos), 'elbos': elbos, 'avg_time':np.mean(times)}

file = 'results/results_lr_local_IW_N{0}_M{1}.json'.format(N,M)
with open(file, 'w') as f:
    json.dump(results_dict, f)
