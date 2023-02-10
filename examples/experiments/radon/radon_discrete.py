import torch as t
import torch.nn as nn
import alan
from torch.distributions.constraints import interval
from torch.distributions.constraint_registry import transform_to
def generate_model(N,M,local,device):
    M = 4
    J = 4
    I = 4
    N = 2
    sizes = {'plate_state': M, 'plate_county':J, 'plate_zipcode':I, 'plate_reading':N}

    def P(tr):
        '''
        Hierarchical Model
        '''

        tr.sample('sigma_beta', alan.Categorical(t.tensor([0.1]*10).to(device)))
        tr.sample('mu_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))
        #state level
        tr.sample('beta', alan.Normal(tr['mu_beta'], tr['sigma_beta'] + 1),plates='plate_state')
        tr.sample('sigma_alpha', alan.Categorical(t.tensor([0.1]*10).to(device)), plates = 'plate_state')

        #county level
        tr.sample('gamma', alan.Categorical(t.tensor([0.1]*10).to(device)), plates = 'plate_county')
        tr.sample('alpha', alan.Normal(tr['beta'] + tr['gamma'] * tr['county_uranium'], tr['sigma_alpha'] + 1), plates = 'plate_county')
        tr.sample('sigma_omega', alan.Categorical(t.tensor([0.1]*10).to(device)), plates='plate_county')

        #zipcode level
        tr.sample('omega', alan.Normal(tr['alpha'], tr['sigma_omega'] + 1), plates='plate_zipcode')
        tr.sample('sigma_obs', alan.Categorical(t.tensor([0.1]*10).to(device)), plates='plate_zipcode')

        #reading level
        tr.sample('psi_int', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)), plates='plate_reading')
        tr.sample('obs', alan.Normal(tr['omega'] + tr['psi_int']*tr['basement'], tr['sigma_obs'] + 1),plates='plate_reading')



    class Q(alan.QModule):
        def __init__(self):
            super().__init__()
            #statevariance
            self.sigma_beta = nn.Parameter(t.rand(10))
            #statemean
            self.mu_beta_mean = nn.Parameter(t.zeros(()))
            self.log_mu_beta_sigma = nn.Parameter(t.zeros(()))
            #countymean
            self.w_beta_mu = nn.Parameter(t.zeros((M,), names=('plate_state',)))
            self.w_beta_sigma = nn.Parameter(t.zeros((M,), names=('plate_state',)))
            self.b_beta_mu = nn.Parameter(t.zeros((M,), names=('plate_state',)))
            self.b_beta_sigma = nn.Parameter(t.zeros((M,), names=('plate_state',)))
            #countyvariance
            self.sigma_alpha = nn.Parameter(t.rand((M,10),names=('plate_state',None)))
            #w
            self.gamma = nn.Parameter(t.rand((J,10),names=('plate_county',None)))
            #zipmean
            self.w_alpha_mu = nn.Parameter(t.zeros((M,J), names=('plate_state', 'plate_county')))
            self.w_alpha_sigma = nn.Parameter(t.zeros((M,J), names=('plate_state', 'plate_county')))
            self.b_alpha_mu = nn.Parameter(t.zeros((M,J), names=('plate_state', 'plate_county')))
            self.b_alpha_sigma = nn.Parameter(t.zeros((M,J), names=('plate_state', 'plate_county')))
            #zipvariance
            self.sigma_omega = nn.Parameter(t.rand((J,10),names=('plate_county',None)))
            #readingmean
            self.w_omega_mu = nn.Parameter(t.zeros((M,J,I), names=('plate_state', 'plate_county', 'plate_zipcode')))
            self.w_omega_sigma = nn.Parameter(t.zeros((M,J,I), names=('plate_state', 'plate_county', 'plate_zipcode')))
            self.b_omega_mu = nn.Parameter(t.zeros((M,J,I), names=('plate_state', 'plate_county', 'plate_zipcode')))
            self.b_omega_sigma = nn.Parameter(t.zeros((M,J,I), names=('plate_state', 'plate_county', 'plate_zipcode')))
            #readingvariance
            self.sigma_obs = nn.Parameter(t.rand((I,10),names=('plate_zipcode',None)))
            #b
            self.psi_int_mu = nn.Parameter(t.zeros((N,), names=('plate_reading',)))
            self.log_psi_int_sigma = nn.Parameter(t.zeros((N,), names=('plate_reading',)))


        def forward(self, tr):
            #state level

            tr.sample('sigma_beta', alan.Categorical(logits=self.sigma_beta))
            tr.sample('mu_beta', alan.Normal(self.mu_beta_mean, self.log_mu_beta_sigma.exp()))
            mean_beta_mu = self.w_beta_mu * tr['mu_beta'] + self.b_beta_mu
            log_beta_sigma = self.w_beta_sigma * tr['sigma_beta'] + self.b_beta_sigma
            tr.sample('beta', alan.Normal(mean_beta_mu, log_beta_sigma.exp()))

            #county level
            tr.sample('gamma', alan.Categorical(logits=self.gamma))
            tr.sample('sigma_alpha', alan.Categorical(logits=self.sigma_alpha))
            mean_alpha_mu = self.w_alpha_mu * tr['beta'] + self.b_alpha_mu + tr['gamma'] * tr['county_uranium']
            log_alpha_sigma = self.w_alpha_sigma * tr['sigma_alpha'] + self.b_alpha_sigma
            tr.sample('alpha', alan.Normal(mean_alpha_mu, log_alpha_sigma.exp()))

            #zipcode level
            tr.sample('sigma_omega', alan.Categorical(logits=self.sigma_omega))
            mean_omega_mu = self.w_omega_mu * tr['alpha'] + self.b_omega_mu
            log_omega_sigma = self.w_omega_sigma * tr['sigma_omega'] + self.b_omega_sigma
            tr.sample('omega', alan.Normal(mean_omega_mu, log_omega_sigma.exp()))

            #reading level
            tr.sample('sigma_obs', alan.Categorical(logits=self.sigma_obs))
            tr.sample('psi_int', alan.Normal(self.psi_int_mu, self.log_psi_int_sigma.exp()))

    covariates = {'basement': t.load('radon/data/train_basement_alongreadings.pt').to(device),
        'county_uranium':t.load('radon/data/county_uranium.pt').rename('plate_state', 'plate_county').to(device)}
    test_covariates = {'basement': t.load('radon/data/test_basement_alongreadings.pt').to(device),
        'county_uranium':t.load('radon/data/county_uranium.pt').rename('plate_state', 'plate_county').to(device)}

    all_covariates = {'basement': t.cat([covariates['basement'],test_covariates['basement']], -1).rename('plate_state', 'plate_county', 'plate_zipcode', 'plate_reading',...),
    'county_uranium':t.load('radon/data/county_uranium.pt').rename('plate_state', 'plate_county').to(device)}
    covariates['basement'] = covariates['basement'].rename('plate_state', 'plate_county', 'plate_zipcode', 'plate_reading',...)

    data = {'obs':t.load('radon/data/train_radon_alongreadings.pt').to(device)}

    test_data = {'obs':t.load('radon/data/test_radon_alongreadings.pt').to(device)}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_state', 'plate_county', 'plate_zipcode', 'plate_reading',...)}
    data['obs'] = data['obs'].rename('plate_state', 'plate_county', 'plate_zipcode', 'plate_reading',...)

    return P, Q, data, covariates, all_data, all_covariates
