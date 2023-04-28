import torch as t
import torch.nn as nn
import alan
import numpy as np

def generate_model(N,M,device,ML=1, run=0, use_data=True):
    M = 3
    J = 3
    I = 30

    sizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':I}

    covariates = {'run_type': t.load('bus_breakdown/data/run_type_train_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
        'bus_company_name': t.load('bus_breakdown/data/bus_company_name_train_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    test_covariates = {'run_type': t.load('bus_breakdown/data/run_type_test_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
        'bus_company_name': t.load('bus_breakdown/data/bus_company_name_test_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    all_covariates = {'run_type': t.cat([covariates['run_type'],test_covariates['run_type']],-3),
        'bus_company_name': t.cat([covariates['bus_company_name'],test_covariates['bus_company_name']],-3)}

    bus_company_name_dim = covariates['bus_company_name'].shape[-1]
    run_type_dim = covariates['run_type'].shape[-1]


    def P(tr, run_type, bus_company_name):
      '''
      Hierarchical Model
      '''

      #Year level
      tr('sigma_beta', alan.Normal(tr.zeros(()), tr.ones(())))
      tr('mu_beta', alan.Normal(tr.zeros(()), tr.ones(())))
      tr('beta', alan.Normal(tr['mu_beta'], tr['sigma_beta'].exp()), plates = 'plate_Year')

      #Borough level
      tr('sigma_alpha', alan.Normal(tr.zeros(()), tr.ones(())), plates = 'plate_Borough')
      tr('alpha', alan.Normal(tr['beta'], tr['sigma_alpha'].exp()))

      #ID level
      tr('log_sigma_phi_psi', alan.Normal(tr.zeros(()), tr.ones(())), plates = 'plate_ID')
      tr('psi', alan.Normal(tr.zeros((run_type_dim,)), tr['log_sigma_phi_psi'].exp()), plates = 'plate_ID')
      tr('phi', alan.Normal(tr.zeros((bus_company_name_dim,)), tr['log_sigma_phi_psi'].exp()), plates = 'plate_ID')
      # tr('theta', alan.Normal(np.log(130) * tr.ones(()), np.log(20) * tr.ones(())))
      # tr('obs', alan.NegativeBinomial(total_count=tr['theta'].exp(), logits=tr['alpha'] + tr['phi'] @ tr['bus_company_name'] + tr['psi'] @ tr['run_type']))
      tr('obs', alan.NegativeBinomial(total_count=10, logits=tr['alpha'] + tr['phi'] @ bus_company_name + tr['psi'] @ run_type))





    class Q(alan.AlanModule):
        def __init__(self):
            super().__init__()
            #sigma_beta
            self.sigma_beta_mean = nn.Parameter(t.zeros(()))
            self.sigma_beta_sigma = nn.Parameter(t.zeros(()))
            #mu_beta
            self.mu_beta_mean = nn.Parameter(t.zeros(()))
            self.log_mu_beta_sigma = nn.Parameter(t.zeros(()))
            #beta
            self.beta_mu = nn.Parameter(t.zeros((M,),names=('plate_Year',)))
            self.log_beta_sigma = nn.Parameter(t.zeros((M,), names=('plate_Year',)))
            #sigma_alpha
            self.sigma_alpha_mean = nn.Parameter(t.zeros((J,), names=('plate_Borough',)))
            self.sigma_alpha_sigma = nn.Parameter(t.zeros((J,), names=('plate_Borough',)))
            #alpha
            self.alpha_mu = nn.Parameter(t.zeros((M,J), names=('plate_Year', 'plate_Borough')))
            self.log_alpha_sigma = nn.Parameter(t.zeros((M,J), names=('plate_Year', 'plate_Borough')))
            #log_sigma_phi_psi logits
            self.log_sigma_phi_psi_mean = nn.Parameter(t.zeros((I,), names=('plate_ID',)))
            self.log_sigma_phi_psi_sigma = nn.Parameter(t.zeros((I,), names=('plate_ID',)))
            #psi
            self.psi_mean = nn.Parameter(t.zeros((I,run_type_dim), names=('plate_ID',None)))
            self.log_psi_sigma = nn.Parameter(t.zeros((I,run_type_dim), names=('plate_ID',None)))
            #phi
            self.phi_mean = nn.Parameter(t.zeros((I,bus_company_name_dim), names=('plate_ID',None)))
            self.log_phi_sigma = nn.Parameter(t.zeros((I,bus_company_name_dim), names=('plate_ID',None)))
            #theta
            # self.theta_mean = nn.Parameter(t.zeros(()))
            # self.log_theta_sigma = nn.Parameter(t.zeros(()))


        def forward(self, tr, run_type, bus_company_name):
            #Year level

            tr('sigma_beta', alan.Normal(self.sigma_beta_mean, self.sigma_beta_sigma.exp()))
            tr('mu_beta', alan.Normal(self.mu_beta_mean, self.log_mu_beta_sigma.exp()))
            tr('beta', alan.Normal(self.beta_mu, self.log_beta_sigma.exp()))

            #Borough level
            tr('sigma_alpha', alan.Normal(self.sigma_alpha_mean, self.sigma_alpha_sigma.exp()))
            tr('alpha', alan.Normal(self.alpha_mu, self.log_alpha_sigma.exp()))

            #ID level
            tr('log_sigma_phi_psi', alan.Normal(self.log_sigma_phi_psi_mean, self.log_sigma_phi_psi_sigma.exp()))
            tr('psi', alan.Normal(self.psi_mean, self.log_psi_sigma.exp()))
            tr('phi', alan.Normal(self.phi_mean, self.log_phi_sigma.exp()))
            # tr('theta', alan.Normal(self.theta_mean, self.log_theta_sigma.exp()))

    if use_data:
        data = {'obs':t.load('bus_breakdown/data/delay_train_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
        test_data = {'obs':t.load('bus_breakdown/data/delay_test_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
        all_data = {'obs': t.cat([data['obs'],test_data['obs']],-2)}
    else:
        model = alan.Model(P, Q())
        data_prior = model.sample_prior(platesizes = sizes, inputs = covariates)
        data_prior_test = model.sample_prior(platesizes = sizes, inputs = test_covariates)
        data = data_prior
        test_data = data_prior_test
        all_data = {'obs': t.cat([data['obs'],test_data['obs']], -2)}

    return P, Q, data, covariates, test_data, test_covariates, all_data, all_covariates, sizes
