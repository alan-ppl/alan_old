import torch as t
import torch.nn as nn
import alan
from torch.distributions.constraints import interval
from torch.distributions.constraint_registry import transform_to
def generate_model(N,M,local,device):
    M = 4
    J = 4
    I = 2
    N = 4
    sizes = {'plate_state': M, 'plate_county':J, 'plate_zipcode':I, 'plate_reading':N}

    def P(tr):
      '''
      Hierarchical Model
      '''

      tr.sample('sigma_beta', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)))
      tr.sample('mu_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))
      #state level
      tr.sample('beta', alan.Normal(tr['mu_beta'], tr['sigma_beta']),plates='plate_state')
      tr.sample('sigma_alpha', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)), plates = 'plate_state')

      #county level
      tr.sample('gamma', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)), plates = 'plate_county')
      tr.sample('alpha', alan.Normal(tr['beta'] + tr['gamma'] * tr['county_uranium'], tr['sigma_alpha']), plates = 'plate_county')
      tr.sample('sigma_omega', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)), plates='plate_county')

      #zipcode level

      tr.sample('omega', alan.Normal(tr['alpha'], tr['sigma_omega']), plates='plate_zipcode')
      tr.sample('sigma_obs', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)), plates='plate_zipcode')

      #reading level

      tr.sample('psi_int', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)), plates='plate_reading')
      tr.sample('obs', alan.Normal(tr['omega'] + tr['psi_int']*tr['basement'], tr['sigma_obs']),plates='plate_reading')



    class Q(alan.QModule):
        def __init__(self):
            super().__init__()
            #statevariance
            self.sigma_beta_p1 = nn.Parameter(t.tensor(0.00001))
            self.sigma_beta_p2 = nn.Parameter(t.tensor(0.00001))
            self.sigma_beta_mid =  nn.Parameter(t.tensor(5.0))
            #statemean
            self.mu_beta_mean = nn.Parameter(t.zeros(()))
            self.log_mu_beta_sigma = nn.Parameter(t.zeros(()))

            #countymean
            self.beta_mu = nn.Parameter(t.zeros((M,),names=('plate_state',)))
            self.log_beta_sigma = nn.Parameter(t.zeros((M,), names=('plate_state',)))
            #countyvariance
            self.sigma_alpha_p1 = nn.Parameter(t.tensor([0.00001]*M, names=('plate_state',)))
            self.sigma_alpha_p2 = nn.Parameter(t.tensor([0.00001]*M, names=('plate_state',)))
            self.sigma_alpha_mid = nn.Parameter(t.tensor([5.0]*M, names=('plate_state',)))

            #w
            self.gamma_p1 = nn.Parameter(t.tensor([0.00001]*J, names=('plate_county',)))
            self.gamma_p2 = nn.Parameter(t.tensor([0.00001]*J, names=('plate_county',)))
            self.gamma_mid = nn.Parameter(t.tensor([5.0]*J, names=('plate_county',)))
            #zipmean
            self.alpha_mu = nn.Parameter(t.zeros((M,J), names=('plate_state', 'plate_county')))
            self.log_alpha_sigma = nn.Parameter(t.zeros((M,J), names=('plate_state', 'plate_county')))
            #zipvariance
            self.sigma_omega_p1 = nn.Parameter(t.tensor([0.00001] * J, names=('plate_county',)))
            self.sigma_omega_p2 = nn.Parameter(t.tensor([0.00001] * J, names=('plate_county',)))
            self.sigma_omega_mid = nn.Parameter(t.tensor([5.0] * J, names=('plate_county',)))

            #readingmean
            self.omega_mu = nn.Parameter(t.zeros((M,J,I), names=('plate_state', 'plate_county', 'plate_zipcode')))
            self.log_omega_sigma = nn.Parameter(t.zeros((M,J,I), names=('plate_state', 'plate_county', 'plate_zipcode')))

            #readingvariance
            self.sigma_obs_p1 = nn.Parameter(t.tensor([0.00001] * I, names=('plate_zipcode',)))
            self.sigma_obs_p2 = nn.Parameter(t.tensor([0.00001] * I, names=('plate_zipcode',)))
            self.sigma_obs_mid = nn.Parameter(t.tensor([5.0] * I, names=('plate_zipcode',)))
            #b
            self.psi_int_mu = nn.Parameter(t.zeros((N), names=('plate_reading',)))
            self.log_psi_int_sigma = nn.Parameter(t.zeros((N), names=('plate_reading',)))

            self.high = t.tensor(10.0 - 1e-6).to(device)
            self.low = t.tensor(0.0).to(device)

        def forward(self, tr):
            #state level
            # sigma_beta_low = t.max(self.low, self.sigma_beta_low.exp())
            # sigma_beta_high = t.min(self.high, self.sigma_beta_high.exp())
            sigma_beta_mid = t.max(self.low, self.sigma_beta_mid)
            sigma_beta_low= t.max(self.low, sigma_beta_mid - self.sigma_beta_p1.exp())
            sigma_beta_high= t.min(self.high, sigma_beta_mid + self.sigma_beta_p2.exp())
            print(sigma_beta_mid)
            print(sigma_beta_low)
            # print('low')
            # print(self.sigma_beta_low)
            # print(sigma_beta_low)
            # print('high')
            # print(self.sigma_beta_high)
            # print(sigma_beta_high)

            tr.sample('sigma_beta', alan.Uniform(sigma_beta_low, sigma_beta_high), multi_sample=False if local else True)
            tr.sample('mu_beta', alan.Normal(self.mu_beta_mean, self.log_mu_beta_sigma.exp()), multi_sample=False if local else True)
            tr.sample('beta', alan.Normal(self.beta_mu, self.log_beta_sigma.exp()), multi_sample=False if local else True)

            #county level
            gamma_mid = t.max(self.low, self.gamma_mid)
            gamma_low= t.max(self.low, gamma_mid - self.gamma_p1.exp())
            gamma_high= t.min(self.high, gamma_mid + self.gamma_p2.exp())

            sigma_alpha_mid = t.max(self.low, self.sigma_alpha_mid)
            sigma_alpha_low= t.max(self.low, sigma_alpha_mid - self.sigma_alpha_p1.exp())
            sigma_alpha_high= t.min(self.high, sigma_alpha_mid + self.sigma_alpha_p2.exp())

            tr.sample('gamma', alan.Uniform(gamma_low, gamma_high), multi_sample=False if local else True)
            tr.sample('sigma_alpha', alan.Uniform(sigma_alpha_low, sigma_alpha_high), multi_sample=False if local else True)
            tr.sample('alpha', alan.Normal(self.alpha_mu, self.log_alpha_sigma.exp()), multi_sample=False if local else True)

            #zipcode level
            sigma_omega_mid = t.max(self.low, self.sigma_omega_mid)
            sigma_omega_low= t.max(self.low, sigma_omega_mid - self.sigma_omega_p1.exp())
            sigma_omega_high= t.min(self.high, sigma_omega_mid + self.sigma_omega_p2.exp())
            tr.sample('sigma_omega', alan.Uniform(sigma_omega_low, sigma_omega_high), multi_sample=False if local else True)
            tr.sample('omega', alan.Normal(self.omega_mu, self.log_omega_sigma.exp()))

            #reading level
            sigma_obs_mid = t.max(self.low, self.sigma_obs_mid)
            sigma_obs_low= t.max(self.low, sigma_obs_mid - self.sigma_obs_p1.exp())
            sigma_obs_high= t.min(self.high, sigma_obs_mid + self.sigma_obs_p2.exp())
            tr.sample('sigma_obs', alan.Uniform(sigma_obs_low, sigma_obs_high))
            tr.sample('psi_int', alan.Normal(self.psi_int_mu, self.log_psi_int_sigma.exp()))

    covariates = {'basement': t.load('radon/data/train_basement_alongzipcodes.pt').to(device),
        'county_uranium':t.load('radon/data/county_uranium.pt').rename('plate_state', 'plate_county').to(device)}
    test_covariates = {'basement': t.load('radon/data/test_basement_alongzipcodes.pt').to(device),
        'county_uranium':t.load('radon/data/county_uranium.pt').rename('plate_state', 'plate_county').to(device)}

    all_covariates = {'basement': t.cat([covariates['basement'],test_covariates['basement']], -1).rename('plate_state', 'plate_county', 'plate_zipcode', 'plate_reading',...),
    'county_uranium':t.load('radon/data/county_uranium.pt').rename('plate_state', 'plate_county').to(device)}
    covariates['basement'] = covariates['basement'].rename('plate_state', 'plate_county', 'plate_zipcode', 'plate_reading',...)

    data = {'obs':t.load('radon/data/train_radon_alongzipcodes.pt').to(device)}

    test_data = {'obs':t.load('radon/data/test_radon_alongzipcodes.pt').to(device)}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_state', 'plate_county', 'plate_zipcode', 'plate_reading',...)}
    data['obs'] = data['obs'].rename('plate_state', 'plate_county', 'plate_zipcode', 'plate_reading',...)

    return P, Q, data, covariates, all_data, all_covariates
