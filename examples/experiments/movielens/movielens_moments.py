import torch as t
import torch.nn as nn
import alan
import time
import numpy as np
import json
from alan.experiment_utils import seed_torch



device = t.device("cuda" if t.cuda.is_available() else "cpu")


Ns = [5,10]
Ms = [10,50,150,300]

for M in Ms:
    for N in Ns:
        sizes = {'plate_1':M, 'plate_2':N}
        d_z = 18
        seed_torch(0)
        def P(tr):
          '''
          Heirarchical Model
          '''

          tr.sample('mu_z', alan.Normal(t.zeros((d_z,)).to(device), t.ones((d_z,)).to(device)))
          tr.sample('psi_z', alan.Normal(t.zeros((d_z,)).to(device), t.ones((d_z,)).to(device)))

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

        covariates = {'x':t.load('data/weights_{0}_{1}.pt'.format(N,M)).to(device)}
        test_covariates = {'x':t.load('data/test_weights_{0}_{1}.pt'.format(N,M)).to(device)}
        all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
        covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)


        data = {'obs':t.load('data/data_y_{0}_{1}.pt'.format(N, M)).to(device)}
        test_data = {'obs':t.load('data/test_data_y_{0}_{1}.pt'.format(N, M)).to(device)}
        all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
        data['obs'] = data['obs'].rename('plate_1','plate_2')

        model = alan.Model(P, Q(), data, covariates)
        model.to(device)
        
        Ks = [1,3,10,30]# 3000]

        elbos = {k:[] for k in Ks}
        elbos_global = {k:[] for k in Ks}
        times = {k:[] for k in Ks}
        times_global = {k:[] for k in Ks}

        num_runs = 100

        for k in Ks:
            for i in range(num_runs):
                start = time.time()
                elbos[k].append(model.elbo(k).item())
                end = time.time()
                times[k].append(end-start)
                start = time.time()
                elbos_global[k].append(model.elbo_global(k).item())
                end = time.time()
                times_global[k].append(end-start)

            elbos[k] = {'mean':np.mean(elbos[k]), 'std_err':np.std(elbos[k])/np.sqrt(num_runs)}
            elbos_global[k] = {'mean':np.mean(elbos_global[k]), 'std_err':np.std(elbos_global[k])/np.sqrt(num_runs)}
            times[k] = {'mean':np.mean(times[k]), 'std_err':np.std(times[k])/np.sqrt(num_runs)}
            times_global[k] = {'mean':np.mean(times_global[k]), 'std_err':np.std(times_global[k])/np.sqrt(num_runs)}

        file = 'results/movielens_moments_elbo_N{0}_M{1}.json'.format(N,M)
        with open(file, 'w') as f:
            json.dump(elbos, f)

        file = 'results/movielens_moments_times_N{0}_M{1}.json'.format(N,M)
        with open(file, 'w') as f:
            json.dump(times, f)

        file = 'results/movielens_moments_elbo_global_N{0}_M{1}.json'.format(N,M)
        with open(file, 'w') as f:
            json.dump(elbos_global, f)

        file = 'results/movielens_moments_times_global_N{0}_M{1}.json'.format(N,M)
        with open(file, 'w') as f:
            json.dump(times_global, f)
