import torch as t
import torch.nn as nn
import alan


import json
import numpy as np
import itertools
from torch.distributions.constraint_registry import transform_to
from torch.distributions.constraints import half_open_interval
import random
import time

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)

seed_torch(0)

print('...', flush=True)

M = 2
J = 2
I = 4
N = 4

device = t.device("cuda" if t.cuda.is_available() else "cpu")

results_dict = {}

Ks = [1,5,10,15]
# Ns = [10,30]
# Ms = [10,50,100]


sizes = {'plate_state': M, 'plate_county':J, 'plate_zipcode':I, 'plate_reading':N}
x = {'basement': t.load('basement.pt').rename('plate_state', 'plate_county', 'plate_zipcode', 'plate_reading',...).to(device),
    'county_uranium':t.load('county_uranium.pt').rename('plate_state', 'plate_county').to(device)}

def P(tr):
  '''
  Hierarchical Model
  '''

  #state level
  tr.sample('sigma_beta', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)), group='local')
  tr.sample('mu_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)), group='local')
  tr.sample('beta', alan.Normal(tr['mu_beta'], tr['sigma_beta']), plates = 'plate_state')

  #county level
  tr.sample('gamma', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)), group='local')
  tr.sample('sigma_alpha', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)), group='local')

  tr.sample('alpha', alan.Normal(tr['beta'] + tr['gamma'] * tr['county_uranium'], tr['sigma_alpha']), group='local')

  #zipcode level
  tr.sample('sigma_omega', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)), group='local')
  tr.sample('omega', alan.Normal(tr['alpha'], tr['sigma_omega']), plates='plate_zipcode')

  #reading level
  tr.sample('sigma_obs', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)), group='local')
  tr.sample('psi_int', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)), group='local')
  tr.sample('obs', alan.Normal(tr['omega'] + tr['psi_int']*tr['basement'], tr['sigma_obs']))



class Q(alan.QModule):
    def __init__(self):
        super().__init__()
        #sigma_beta
        self.sigma_beta_low = nn.Parameter(t.tensor(0.00001).log())
        self.sigma_beta_high = nn.Parameter(t.tensor(9.9999).log())
        #mu_beta
        self.mu_beta_mean = nn.Parameter(t.zeros(()))
        self.log_mu_beta_sigma = nn.Parameter(t.zeros(()))
        #beta
        self.beta_mu = nn.Parameter(t.zeros((M)), names=('plate_state'))
        self.log_beta_sigma = nn.Parameter(t.zeros((M)), names=('plate_state'))
        #gamma
        self.gamma_low = nn.Parameter(t.tensor(0.00001).log())
        self.gamma_high = nn.Parameter(t.tensor(9.9999).log())
        #sigma_alpha
        self.sigma_alpha_low = nn.Parameter(t.tensor(0.00001).log())
        self.sigma_alpha_high = nn.Parameter(t.tensor(9.9999).log())
        #alpha
        self.alpha_mu = nn.Parameter(t.zeros((M,J)), names=('plate_state', 'plate_county'))
        self.log_alpha_sigma = nn.Parameter(t.zeros((M,J)), names=('plate_state', 'plate_county'))
        #sigma_omega
        self.sigma_omega_low = nn.Parameter(t.tensor(0.00001).log())
        self.sigma_omega_high = nn.Parameter(t.tensor(9.9999).log())
        #omega
        self.omega_mu = nn.Parameter(t.zeros((M,J,I)), names=('plate_state', 'plate_county', 'plate_zipcode'))
        self.log_omega_sigma = nn.Parameter(t.zeros((M,J,I)), names=('plate_state', 'plate_county', 'plate_zipcode'))

        #sigma_obs
        self.sigma_obs_low = nn.Parameter(t.tensor(0.00001).log())
        self.sigma_obs_high = nn.Parameter(t.tensor(9.9999).log())

        #beta_int
        self.psi_int_mu = nn.Parameter(t.zeros(()))
        self.log_psi_int_sigma = nn.Parameter(t.zeros(()))

        self.high = t.tensor(10.0).to(device)
        self.low = t.tensor(0.0).to(device)

    def forward(self, tr):
        #state level
        sigma_beta_low = t.max(self.low, self.sigma_beta_low.exp())
        sigma_beta_high = t.min(self.high, self.sigma_beta_high.exp())

        tr.sample('sigma_beta', alan.Uniform(sigma_beta_low, sigma_beta_high))
        tr.sample('mu_beta', alan.Normal(self.mu_beta_mean, self.log_mu_beta_sigma.exp()))
        tr.sample('beta', alan.Normal(self.beta_mu, self.log_beta_sigma.exp()))

        #county level
        gamma_low = t.max(self.low, self.gamma_low.exp())
        gamma_high = t.min(self.high, self.gamma_high.exp())

        sigma_alpha_low = t.max(self.low, self.sigma_alpha_low.exp())
        sigma_alpha_high = t.min(self.high, self.sigma_alpha_high.exp())
        tr.sample('gamma', alan.Uniform(gamma_low, gamma_high))
        tr.sample('sigma_alpha', alan.Uniform(sigma_alpha_low, sigma_alpha_high))
        tr.sample('alpha', alan.Normal(self.alpha_mu, self.log_alpha_sigma.exp()))

        #zipcode level
        sigma_omega_low = t.max(self.low, self.sigma_omega_low.exp())
        sigma_omega_high = t.min(self.high, self.sigma_omega_high.exp())
        tr.sample('sigma_omega', alan.Uniform(sigma_omega_low, sigma_omega_high))
        tr.sample('omega', alan.Normal(self.omega_mu, self.log_omega_sigma.exp()))

        #reading level
        sigma_obs_low = t.max(self.low, self.sigma_obs_low.exp())
        sigma_obs_high = t.min(self.high, self.sigma_obs_high.exp())
        tr.sample('sigma_obs', alan.Uniform(sigma_obs_low, sigma_obs_high))
        tr.sample('psi_int', alan.Normal(self.psi_int_mu, self.log_psi_int_sigma.exp()))

data_y = {'obs':t.load('radon.pt').rename('plate_state', 'plate_county', 'plate_zipcode', 'plate_reading',...).to(device)}

for K in Ks:
    print(K)
    results_dict[K] = results_dict.get(K, {})
    elbos = []
    times = []
    for i in range(5):
        per_seed_elbos = []
        start = time.time()
        seed_torch(i)

        model = alan.Model(P, Q(), data_y | x)
        model.to(device)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)
        scheduler = t.optim.lr_scheduler.StepLR(opt, step_size=20000, gamma=0.1)


        for j in range(100000):
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
    results_dict[K] = {'lower_bound':np.mean(elbos),'std':np.std(elbos), 'avg_time':np.mean(times)}

file = 'results/results_unif_LIW.json'
with open(file, 'w') as f:
    json.dump(results_dict, f)
