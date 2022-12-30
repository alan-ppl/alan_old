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


np.random.seed(0)

M = args.M
N = args.N


sizes = {'plate_1':M, 'plate_2':N}

x = {'x':t.load('data/weights_{0}_{1}.pt'.format(N,M)).rename('plate_1','plate_2',...).to(device)}
d_z = 18
def P(tr):
  '''
  Heirarchical Model
  '''

  tr.sample('mu_z', alan.Normal(t.zeros((d_z,)).to(device), t.ones((d_z,)).to(device)), group='local')
  tr.sample('psi_z', alan.Normal(t.zeros((d_z,)).to(device), t.ones((d_z,)).to(device)), group='local')

  tr.sample('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()), plates='plate_1')

  tr.sample('obs', alan.Bernoulli(logits = tr['z'] @ tr['x']))



class Q(alan.QModule):
    def __init__(self):
        super().__init__()
        #mu_z
        self.m_mu_z = nn.Parameter(t.zeros((d_z,)))
        self.log_theta_mu_z = nn.Parameter(t.zeros((d_z,)))
        #psi_z
        self.m_psi_z = nn.Parameter(t.zeros((d_z,)))
        self.log_theta_psi_z = nn.Parameter(t.zeros((d_z,)))

        #z
        self.mu = nn.Parameter(t.zeros((M,d_z), names=('plate_1',None)))
        self.log_sigma = nn.Parameter(t.zeros((M,d_z), names=('plate_1',None)))


    def forward(self, tr):
        tr.sample('mu_z', alan.Normal(self.m_mu_z, self.log_theta_mu_z.exp()))
        tr.sample('psi_z', alan.Normal(self.m_psi_z, self.log_theta_psi_z.exp()))

        tr.sample('z', alan.Normal(self.mu, self.log_sigma.exp()))





data_y = {'obs':t.load('data/data_y_{0}_{1}.pt'.format(N, M)).rename('plate_1','plate_2').to(device)}

for K in Ks:
    print(K,M,N)
    results_dict[N] = results_dict.get(N, {})
    results_dict[N][M] = results_dict[N].get(M, {})
    results_dict[N][M][K] = results_dict[N][M].get(K, {})
    elbos = []
    times = []
    for i in range(5):
        per_seed_elbos = []
        seed_torch(i)
        start = time.time()

        model = alan.Model(P, Q(), data_y | x)
        model.to(device)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)


        for i in range(50000):
            opt.zero_grad()
            elbo = model.elbo(K=K)
            (-elbo).backward()
            opt.step()
            per_seed_elbos.append(elbo.item())
            if 0 == i%1000:
                print("Iteration: {0}, ELBO: {1:.2f}".format(i,elbo.item()))

        elbos.append(np.mean(per_seed_elbos[-50:]))
        times.append(time.time() - start)
    results_dict[N][M][K] = {'lower_bound':np.mean(elbos),'std':np.std(elbos), 'elbos': elbos, 'avg_time':np.mean(times)}

file = 'results/movielens_results_local_IW_N{0}_M{1}.json'.format(N,M)
with open(file, 'w') as f:
    json.dump(results_dict, f)
