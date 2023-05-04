import torch as t
import torch.nn as nn
import alan
import matplotlib.pyplot as plt
import numpy as np
from alan.experiment_utils import seed_torch
import time

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

# Model
N=20
M=450
sizes = {'plate_1':M, 'plate_2':N}
d_z = 18
def P(tr, x):
  '''
  Heirarchical Model
  '''

  tr('mu_z', alan.Normal(tr.zeros((d_z,)), 0.25*tr.ones((d_z,))))
  tr('psi_z', alan.Normal(tr.zeros((d_z,)), 0.25*tr.ones((d_z,))))

  tr('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()), plates='plate_1')

  tr('obs', alan.Bernoulli(logits = tr['z'] @ x))


class Q(alan.AlanModule):
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


    def forward(self, tr, x):
        tr('mu_z', alan.Normal(self.m_mu_z, self.log_theta_mu_z.exp()))
        tr('psi_z', alan.Normal(self.m_psi_z, self.log_theta_psi_z.exp()))

        tr('z', alan.Normal(self.mu, self.log_sigma.exp()))

# DATA
def get_data(run):
    covariates = {'x':t.load('data/weights_20_450_{}.pt'.format(run))}
    test_covariates = {'x':t.load('data/test_weights_20_450_{}.pt'.format(run))}
    all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
    covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)
    test_covariates['x'] = test_covariates['x'].rename('plate_1','plate_2',...)


    data = {'obs':t.load('data/data_y_{0}_{1}_{2}.pt'.format(N, M,run))}
    test_data = {'obs':t.load('data/test_data_y_{0}_{1}_{2}.pt'.format(N, M,run))}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
    data['obs'] = data['obs'].rename('plate_1','plate_2')
    test_data['obs'] = test_data['obs'].rename('plate_1','plate_2')

    return data, covariates, test_data, test_covariates, all_data, all_covariates



per_seed_elbo = np.zeros((5,10000), dtype=np.float32)
times = np.zeros((5,10000), dtype=np.float32)
for i in range(5):
    print(f'Run: {i+1}')
    seed_torch(i)
    model = alan.Model(P, Q())
    model.to(device)

    opt = t.optim.Adam(model.parameters(), lr=0.03)

    data, covariates, test_data, test_covariates, all_data, all_covariates = get_data(i)
    for j in range(10000):
        if t.cuda.is_available():
            t.cuda.synchronize()
        start = time.time()
        opt.zero_grad()
        sample = model.sample_perm(1, data=data, inputs=covariates, reparam=True, device=device)
        elbo = sample.elbo()
        per_seed_elbo[i,j] = elbo.item()
        (-elbo).backward()
        opt.step()
        if t.cuda.is_available():
            t.cuda.synchronize()
        times[i,j] = (time.time() - start)

        if j % 100 == 0:
            print("Iteration: {0}, ELBO: {1:.2f}".format(j,elbo))



#Plotting
fig, ax = plt.subplots(1,1, figsize=(5.5, 5.5), constrained_layout=True)
elbos = per_seed_elbo.mean(axis=0)
elbos_std = per_seed_elbo.std(axis=0) / np.sqrt(5)
times = times.mean(axis=0).cumsum(axis=0)

ax.errorbar(times.tolist(),elbos.tolist(), linewidth=0.55, markersize = 0.75, fmt='-',)
ax.set_ylim(-7000,-5800)
fig.savefig('elbo.pdf')
