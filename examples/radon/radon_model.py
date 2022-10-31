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

t.manual_seed(0)
# parser = argparse.ArgumentParser(description='Run the Heirarchical regression task.')
#
# parser.add_argument('N', type=int,
#                     help='Scale of experiment')
# parser.add_argument('M', type=int,
#                     help='Number of groups')
#
# args = parser.parse_args()

print('...', flush=True)


device = t.device("cuda" if t.cuda.is_available() else "cpu")

results_dict = {}

Ks = [1,5,10,15]
# Ns = [10,30]
# Ms = [10,50,100]

plate_state, plate_county, plate_zipcode, plate_reading = dims(5 , [2,2,2,4])
basement = t.load('basement.pt').to(device)
county_uranium = t.load('county_uranium.pt').to(device)
def P(tr):
  '''
  Heirarchical Model
  '''

  #state level
  tr['sigma_beta'] = tpp.Uniform(t.tensor([0.0]).to(device), t.tensor([100.0]).to(device))
  tr['mu_beta'] = tpp.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device))
  tr['beta'] = tpp.Normal(mu_beta, sigma_beta, sample_dim = plate_state)

  #county level
  tr['gamma'] = tpp.Uniform(t.tensor([0.0]).to(device), t.tensor([100.0]).to(device), sample_dim=plate_county)
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
        self.reg_param("sigma_beta_mu", t.zeros(()))
        self.reg_param("log_sigma_beta_sigma", t.zeros(()))
        #mu_beta
        self.reg_param("mu_beta_mean", t.zeros(()))
        self.reg_param("log_mu_beta_sigma", t.zeros(()))
        #beta
        self.reg_param('beta_mu', t.zeros((2)), [plate_state])
        self.reg_param("log_beta_sigma", t.zeros((2)), [plate_state])
        #gamma
        self.reg_param("gamma_mu", t.zeros((2,2)), [plate_state, plate_county])
        self.reg_param("log_gamma_sigma", t.zeros((2,2)), [plate_state, plate_county])
        #sigma_alpha
        self.reg_param("sigma_alpha_mu", t.zeros(()))
        self.reg_param("log_sigma_alpha_sigma", t.zeros(()))
        #alpha
        self.reg_param("alpha_mu", t.zeros((2,2,2)), [plate_state, plate_county, plate_zipcode])
        self.reg_param("log_alpha_sigma", t.zeros((2,2,2)), [plate_state, plate_county, plate_zipcode])

        #sigma_omega
        self.reg_param("sigma_omega_mu", t.zeros(()))
        self.reg_param("log_sigma_omega_sigma", t.zeros(()))

        #omega
        self.reg_param("omega_mu", t.zeros((2,2,2,4)), [plate_state, plate_county, plate_zipcode])
        self.reg_param("log_omega_sigma", t.zeros((2,2,2,4)), [plate_state, plate_county, plate_zipcode])

    def forward(self, tr):
        tr['mu_z1'] = tpp.Normal(self.m_mu_z1, self.log_theta_mu_z1.exp())
        tr['mu_z2'] = tpp.Normal(self.m_mu_z2, self.log_theta_mu_z2.exp())
        tr['mu_z3'] = tpp.Normal(self.m_mu_z3, self.log_theta_mu_z3.exp())
        tr['mu_z4'] = tpp.Normal(self.m_mu_z4, self.log_theta_mu_z4.exp())
        tr['psi_z'] = tpp.Normal(self.m_psi_z, self.log_theta_psi_z.exp())
        tr['psi_y'] = tpp.Normal(self.m_psi_y, self.log_theta_psi_y.exp())


        tr['z'] = tpp.Normal(self.mu, self.log_sigma.exp())

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
        scheduler = t.optim.lr_scheduler.StepLR(opt, step_size=10000, gamma=0.1)



        dim = tpp.make_dims(P, K)

        for i in range(50000):
            opt.zero_grad()
            elbo = model.elbo(dims=dim)
            (-elbo).backward()
            opt.step()
            scheduler.step()

            if 0 == i%1000:
                print("Iteration: {0}, ELBO: {1:.2f}".format(i,elbo.item()))

        elbos.append(elbo.item())
    results_dict[N][M][K] = {'lower_bound':np.mean(elbos),'std':np.std(elbos), 'elbos': elbos}

file = 'results/results_N{0}_M{1}.json'.format(N,M)
with open(file, 'w') as f:
    json.dump(results_dict, f)
