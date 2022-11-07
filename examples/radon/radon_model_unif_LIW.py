import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from functorch.dim import dims
# import argparse
import json
import numpy as np
import itertools
from torch.distributions.constraint_registry import transform_to
from torch.distributions.constraints import half_open_interval
import random

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)

seed_torch(0)
# parser = argparse.ArgumentParser(description='Run the Heirarchical regression task.')
#
# parser.add_argument('N', type=int,
#                     help='Scale of experiment')
# parser.add_argument('M', type=int,
#                     help='Number of groups')
#
# args = parser.parse_args()

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

plate_state, plate_county, plate_zipcode, plate_reading = dims(4 , [M,J,I,N])
basement = t.load('basement.pt').to(device)[plate_state, plate_county, plate_zipcode]
county_uranium = t.load('county_uranium.pt').to(device)[plate_state, plate_county]
def P(tr):
  '''
  Hierarchical Model
  '''

  #state level
  tr['sigma_beta'] = tpp.Uniform(t.tensor([0.0]).to(device), t.tensor([100.0]).to(device))
  tr['mu_beta'] = tpp.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device))
  tr['beta'] = tpp.Normal(tr['mu_beta'], tr['sigma_beta'], sample_dim = plate_state)

  #county level
  tr['gamma'] = tpp.Uniform(t.tensor([0.0]).to(device), t.tensor([100.0]).to(device))
  tr['sigma_alpha'] = tpp.Uniform(t.tensor([0.0]).to(device), t.tensor([100.0]).to(device))

  tr['alpha'] = tpp.Normal(tr['beta'] + tr['gamma'] * county_uranium, tr['sigma_alpha'])

  #zipcode level
  tr['sigma_omega'] = tpp.Uniform(t.tensor([0.0]).to(device), t.tensor([100.0]).to(device))
  tr['omega'] = tpp.Normal(tr['sigma_alpha'], tr['sigma_omega'], sample_dim=plate_zipcode)

  #reading level
  tr['sigma_obs'] = tpp.Uniform(t.tensor([0.0]).to(device), t.tensor([100.0]).to(device))
  tr['beta_int'] = tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device))
  tr['obs'] = tpp.Normal(tr['omega'] + tr['beta_int']*basement, tr['sigma_obs'], sample_dim=plate_reading)



class Q(tpp.Q_module):
    def __init__(self):
        super().__init__()
        #sigma_beta
        self.reg_param("sigma_beta_low", t.tensor([0.0]))
        self.reg_param("sigma_beta_high", t.tensor([100.0]))
        #mu_beta
        self.reg_param("mu_beta_mean", t.zeros(()))
        self.reg_param("log_mu_beta_sigma", t.zeros(()))
        #beta
        self.reg_param('beta_mu', t.zeros((M)), [plate_state])
        self.reg_param("log_beta_sigma", t.zeros((M)), [plate_state])
        #gamma
        self.reg_param("gamma_low", t.tensor([0.0]))
        self.reg_param("gamma_high", t.tensor([100.0]))
        #sigma_alpha
        self.reg_param("sigma_alpha_low", t.tensor([0.0]))
        self.reg_param("sigma_alpha_high", t.tensor([100.0]))
        #alpha
        self.reg_param("alpha_mu", t.zeros((M,J)), [plate_state, plate_county])
        self.reg_param("log_alpha_sigma", t.zeros((M,J)), [plate_state, plate_county])

        #sigma_omega
        self.reg_param("sigma_omega_low", t.tensor([0.0]))
        self.reg_param("sigma_omega_high", t.tensor([100.0]))

        #omega
        self.reg_param("omega_mu", t.zeros((M,J,I)), [plate_state, plate_county, plate_zipcode])
        self.reg_param("log_omega_sigma", t.zeros((M,J,I)), [plate_state, plate_county, plate_zipcode])

        #sigma_obs
        self.reg_param("sigma_obs_low", t.tensor([0.0]))
        self.reg_param("sigma_obs_high", t.tensor([100.0]))

        #beta_int
        self.reg_param("beta_int_mu", t.zeros(()))
        self.reg_param("log_beta_int_sigma", t.zeros(()))

    def forward(self, tr):
        #state level
        sigma_beta_low = transform_to(half_open_interval(0.0, self.sigma_beta_high))(self.sigma_beta_low)
        sigma_beta_high = transform_to(half_open_interval(sigma_beta_low, 100.0))(self.sigma_beta_high)
        # print(self.sigma_beta_low)
        # print(sigma_beta_low)
        # print(self.sigma_beta_high)
        # print(sigma_beta_high)
        tr['sigma_beta'] = tpp.Uniform(sigma_beta_low, sigma_beta_high, sample_K=False)
        tr['mu_beta'] = tpp.Normal(self.mu_beta_mean, self.log_mu_beta_sigma.exp(), sample_K=False)
        tr['beta'] = tpp.Normal(self.beta_mu, self.log_beta_sigma.exp())

        #county level
        gamma_low = transform_to(half_open_interval(0.0, self.gamma_high - 1.0))(self.gamma_low)
        gamma_high = transform_to(half_open_interval(gamma_low, 100.0))(self.gamma_high)

        sigma_alpha_low = transform_to(half_open_interval(0.0, self.sigma_alpha_high - 1.0))(self.sigma_alpha_low)
        sigma_alpha_high = transform_to(half_open_interval(sigma_alpha_low, 100.0))(self.sigma_alpha_high)
        tr['gamma'] = tpp.Uniform(gamma_low, gamma_high, sample_K=False)
        tr['sigma_alpha'] = tpp.Uniform(sigma_alpha_low, sigma_alpha_high, sample_K=False)
        tr['alpha'] = tpp.Normal(self.alpha_mu, self.log_alpha_sigma.exp())

        #zipcode level
        sigma_omega_low = transform_to(half_open_interval(0.0, self.sigma_omega_high - 1.0))(self.sigma_omega_low)
        sigma_omega_high = transform_to(half_open_interval(sigma_omega_low, 100.0))(self.sigma_omega_high)
        tr['sigma_omega'] = tpp.Uniform(sigma_omega_low, sigma_omega_high, sample_K=False)
        tr['omega'] = tpp.Normal(self.omega_mu, self.log_omega_sigma.exp())

        #reading level
        sigma_obs_low = transform_to(half_open_interval(0.0, self.sigma_obs_high - 1.0))(self.sigma_obs_low)
        sigma_obs_high = transform_to(half_open_interval(sigma_obs_low, 100.0))(self.sigma_obs_high)
        tr['sigma_obs'] = tpp.Uniform(sigma_obs_low, sigma_obs_high, sample_K=False)
        tr['beta_int'] = tpp.Normal(self.beta_int_mu, self.log_beta_int_sigma.exp(), sample_K=False)

data_y = {'obs':t.load('radon.pt')[plate_state, plate_county, plate_zipcode, plate_reading].to(device)}

for K in Ks:
    print(K)
    results_dict[K] = results_dict.get(K, {})
    elbos = []

    for i in range(5):

        seed_torch(i)

        model = tpp.Model(P, Q(), data_y)
        model.to(device)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)
        scheduler = t.optim.lr_scheduler.StepLR(opt, step_size=10000, gamma=0.1)



        dim = tpp.make_dims(P, K, exclude=['sigma_beta', 'mu_beta', 'gamma', 'sigma_alpha', 'sigma_omega', 'sigma_obs', 'beta_int'])

        for i in range(100000):
            opt.zero_grad()
            elbo = model.elbo(dims=dim)
            (-elbo).backward()
            opt.step()
            scheduler.step()

            if 0 == i%1000:
                print("Iteration: {0}, ELBO: {1:.2f}".format(i,elbo.item()))

        elbos.append(elbo.item())
    results_dict[K] = {'lower_bound':np.mean(elbos),'std':np.std(elbos), 'elbos': elbos}

file = 'results/results_unif_LIW.json'
with open(file, 'w') as f:
    json.dump(results_dict, f)