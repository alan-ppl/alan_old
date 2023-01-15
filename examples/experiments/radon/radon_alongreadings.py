import torch as t
import torch.nn as nn
import alan

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
            self.sigma_beta_low = nn.Parameter(t.tensor(0.00001).log())
            self.sigma_beta_high = nn.Parameter(t.tensor(9.9999).log())
            #statemean
            self.mu_beta_mean = nn.Parameter(t.zeros(()))
            self.log_mu_beta_sigma = nn.Parameter(t.zeros(()))

            #countymean
            self.beta_mu = nn.Parameter(t.zeros((M,),names=('plate_state',)))
            self.log_beta_sigma = nn.Parameter(t.zeros((M,), names=('plate_state',)))
            #countyvariance
            self.sigma_alpha_low = nn.Parameter(t.tensor([0.00001]*M, names=('plate_state',)).log())
            self.sigma_alpha_high = nn.Parameter(t.tensor([9.9999]*M, names=('plate_state',)).log())

            #w
            self.gamma_low = nn.Parameter(t.tensor([0.00001]*J, names=('plate_county',)).log())
            self.gamma_high = nn.Parameter(t.tensor([9.9999]*J, names=('plate_county',)).log())
            #zipmean
            self.alpha_mu = nn.Parameter(t.zeros((M,J), names=('plate_state', 'plate_county')))
            self.log_alpha_sigma = nn.Parameter(t.zeros((M,J), names=('plate_state', 'plate_county')))
            #zipvariance
            self.sigma_omega_low = nn.Parameter(t.tensor([0.00001] * J, names=('plate_county',)).log())
            self.sigma_omega_high = nn.Parameter(t.tensor([9.9999] * J, names=('plate_county',)).log())

            #readingmean
            self.omega_mu = nn.Parameter(t.zeros((M,J,I), names=('plate_state', 'plate_county', 'plate_zipcode')))
            self.log_omega_sigma = nn.Parameter(t.zeros((M,J,I), names=('plate_state', 'plate_county', 'plate_zipcode')))

            #readingvariance
            self.sigma_obs_low = nn.Parameter(t.tensor([0.00001] * I, names=('plate_zipcode',)).log())
            self.sigma_obs_high = nn.Parameter(t.tensor([9.9999] * I, names=('plate_zipcode',)).log())

            #b
            self.psi_int_mu = nn.Parameter(t.zeros((N), names=('plate_reading',)))
            self.log_psi_int_sigma = nn.Parameter(t.zeros((N), names=('plate_reading',)))

            self.high = t.tensor(10.0).to(device)
            self.low = t.tensor(0.0 + 1e-5).to(device)

        def forward(self, tr):
            #state level
            # sigma_beta_low = t.max(self.low, self.sigma_beta_low.exp())
            # sigma_beta_high = t.min(self.high, self.sigma_beta_high.exp())
            sigma_beta_low_interval = interval(self.low, self.high)
            sigma_beta_low = transform_to(sigma_beta_low_interval)(self.sigma_beta_low)

            sigma_beta_high_interval = interval(sigma_beta_low, self.high)
            sigma_beta_high = transform_to(sigma_beta_high_interval)(self.sigma_beta_high)


            tr.sample('sigma_beta', alan.Uniform(sigma_beta_low, sigma_beta_high), multi_sample=False if local else True)
            tr.sample('mu_beta', alan.Normal(self.mu_beta_mean, self.log_mu_beta_sigma.exp()), multi_sample=False if local else True)
            tr.sample('beta', alan.Normal(self.beta_mu, self.log_beta_sigma.exp()), multi_sample=False if local else True)

            #county level
            # gamma_low = t.max(self.low, self.gamma_low.exp())
            # gamma_high = t.min(self.high, self.gamma_high.exp())
            gamma_low_interval = interval(self.low, self.high)
            gamma_low = transform_to(gamma_low_interval)(self.gamma_low)

            gamma_high_interval = interval(gamma_low, self.high)
            gamma_high = transform_to(gamma_high_interval)(self.gamma_high)

            # sigma_alpha_low = t.max(self.low, self.sigma_alpha_low.exp())
            # sigma_alpha_high = t.min(self.high, self.sigma_alpha_high.exp())
            # sigma_alpha_low = t.min(sigma_alpha_low, sigma_alpha_high - 0.001)
            sigma_alpha_low_interval = interval(self.low, self.high)
            sigma_alpha_low = transform_to(sigma_alpha_low_interval)(self.sigma_alpha_low)

            sigma_alpha_high_interval = interval(sigma_alpha_low, self.high)
            sigma_alpha_high = transform_to(sigma_alpha_high_interval)(self.sigma_alpha_high)
            tr.sample('gamma', alan.Uniform(gamma_low, gamma_high), multi_sample=False if local else True)
            tr.sample('sigma_alpha', alan.Uniform(sigma_alpha_low, sigma_alpha_high), multi_sample=False if local else True)
            tr.sample('alpha', alan.Normal(self.alpha_mu, self.log_alpha_sigma.exp()), multi_sample=False if local else True)

            #zipcode level
            # sigma_omega_low = t.max(self.low, self.sigma_omega_low.exp())
            # sigma_omega_high = t.min(self.high, self.sigma_omega_high.exp())
            # sigma_omega_low = t.min(sigma_omega_low, sigma_omega_high - 0.001)
            sigma_omega_low_interval = interval(self.low, self.high)
            sigma_omega_low = transform_to(sigma_omega_low_interval)(self.sigma_omega_low)

            sigma_omega_high_interval = interval(sigma_omega_low, self.high)
            sigma_omega_high = transform_to(sigma_omega_high_interval)(self.sigma_omega_high)
            tr.sample('sigma_omega', alan.Uniform(sigma_omega_low, sigma_omega_high), multi_sample=False if local else True)
            tr.sample('omega', alan.Normal(self.omega_mu, self.log_omega_sigma.exp()))

            #reading level
            # sigma_obs_low = t.max(self.low, self.sigma_obs_low.exp())
            # sigma_obs_high = t.min(self.high, self.sigma_obs_high.exp())
            # sigma_obs_low = t.min(sigma_obs_low, sigma_obs_high - 0.001)
            sigma_obs_low_interval = interval(self.low, self.high)
            sigma_obs_low = transform_to(sigma_obs_low_interval)(self.sigma_obs_low)

            sigma_obs_high_interval = interval(sigma_obs_low, self.high)
            sigma_obs_high = transform_to(sigma_obs_high_interval)(self.sigma_obs_high)
            tr.sample('sigma_obs', alan.Uniform(sigma_obs_low, sigma_obs_high))
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
