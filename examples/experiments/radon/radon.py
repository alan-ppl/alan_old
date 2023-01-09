import torch as t
import torch.nn as nn
import alan

def generate_model(N,M,local,device):
    M = 2
    J = 2
    I = 4
    N = 4
    sizes = {'plate_state': M, 'plate_county':J, 'plate_zipcode':I, 'plate_reading':N}

    def P(tr):
      '''
      Hierarchical Model
      '''

      #state level
      tr.sample('sigma_beta', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)), plates = 'plate_state')
      tr.sample('mu_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)), plates = 'plate_state')
      tr.sample('beta', alan.Normal(tr['mu_beta'], tr['sigma_beta']))

      #county level
      tr.sample('gamma', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)), plates = 'plate_county')
      tr.sample('sigma_alpha', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)), plates = 'plate_county')

      tr.sample('alpha', alan.Normal(tr['beta'] + tr['gamma'] * tr['county_uranium'], tr['sigma_alpha']))

      #zipcode level
      tr.sample('sigma_omega', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)), plates='plate_zipcode')
      tr.sample('omega', alan.Normal(tr['alpha'], tr['sigma_omega']))

      #reading level
      tr.sample('sigma_obs', alan.Uniform(t.tensor(0.0).to(device), t.tensor(10.0).to(device)), plates='plate_reading')
      tr.sample('psi_int', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)), plates='plate_reading')
      tr.sample('obs', alan.Normal(tr['omega'] + tr['psi_int']*tr['basement'], tr['sigma_obs']))



    class Q(alan.QModule):
        def __init__(self):
            super().__init__()
            #sigma_beta
            self.sigma_beta_low = nn.Parameter(t.tensor([0.00001] * M, names=('plate_state',)).log())
            self.sigma_beta_high = nn.Parameter(t.tensor([9.9999] * M, names=('plate_state',)).log())
            #mu_beta
            self.mu_beta_mean = nn.Parameter(t.zeros((M,), names=('plate_state',)))
            self.log_mu_beta_sigma = nn.Parameter(t.zeros((M,), names=('plate_state',)))
            #beta
            self.beta_mu = nn.Parameter(t.zeros((M,),names=('plate_state',)))
            self.log_beta_sigma = nn.Parameter(t.zeros((M,), names=('plate_state',)))
            #gamma
            self.gamma_low = nn.Parameter(t.tensor([0.00001]*J, names=('plate_county',)).log())
            self.gamma_high = nn.Parameter(t.tensor([9.9999]*J, names=('plate_county',)).log())
            #sigma_alpha
            self.sigma_alpha_low = nn.Parameter(t.tensor([0.00001]*J, names=('plate_county',)).log())
            self.sigma_alpha_high = nn.Parameter(t.tensor([9.9999]*J, names=('plate_county',)).log())
            #alpha
            self.alpha_mu = nn.Parameter(t.zeros((M,J), names=('plate_state', 'plate_county')))
            self.log_alpha_sigma = nn.Parameter(t.zeros((M,J), names=('plate_state', 'plate_county')))
            #sigma_omega
            self.sigma_omega_low = nn.Parameter(t.tensor([0.00001] * I, names=('plate_zipcode',)).log())
            self.sigma_omega_high = nn.Parameter(t.tensor([9.9999] * I, names=('plate_zipcode',)).log())
            #omega
            self.omega_mu = nn.Parameter(t.zeros((M,J,I), names=('plate_state', 'plate_county', 'plate_zipcode')))
            self.log_omega_sigma = nn.Parameter(t.zeros((M,J,I), names=('plate_state', 'plate_county', 'plate_zipcode')))

            #sigma_obs
            self.sigma_obs_low = nn.Parameter(t.tensor([0.00001] * N, names=('plate_reading',)).log())
            self.sigma_obs_high = nn.Parameter(t.tensor([9.9999] * N, names=('plate_reading',)).log())

            #beta_int
            self.psi_int_mu = nn.Parameter(t.zeros((N), names=('plate_reading',)))
            self.log_psi_int_sigma = nn.Parameter(t.zeros((N), names=('plate_reading',)))

            self.high = t.tensor(10.0).to(device)
            self.low = t.tensor(0.0).to(device)

        def forward(self, tr):
            #state level
            sigma_beta_low = t.max(self.low, self.sigma_beta_low.exp())
            sigma_beta_high = t.min(self.high, self.sigma_beta_high.exp())

            tr.sample('sigma_beta', alan.Uniform(sigma_beta_low, sigma_beta_high), multi_sample=False if local else None)
            tr.sample('mu_beta', alan.Normal(self.mu_beta_mean, self.log_mu_beta_sigma.exp()), multi_sample=False if local else None)
            tr.sample('beta', alan.Normal(self.beta_mu, self.log_beta_sigma.exp()), multi_sample=False if local else None)

            #county level
            gamma_low = t.max(self.low, self.gamma_low.exp())
            gamma_high = t.min(self.high, self.gamma_high.exp())

            sigma_alpha_low = t.max(self.low, self.sigma_alpha_low.exp())
            sigma_alpha_high = t.min(self.high, self.sigma_alpha_high.exp())
            tr.sample('gamma', alan.Uniform(gamma_low, gamma_high), multi_sample=False if local else None)
            tr.sample('sigma_alpha', alan.Uniform(sigma_alpha_low, sigma_alpha_high), multi_sample=False if local else None)
            tr.sample('alpha', alan.Normal(self.alpha_mu, self.log_alpha_sigma.exp()), multi_sample=False if local else None)

            #zipcode level
            sigma_omega_low = t.max(self.low, self.sigma_omega_low.exp())
            sigma_omega_high = t.min(self.high, self.sigma_omega_high.exp())
            tr.sample('sigma_omega', alan.Uniform(sigma_omega_low, sigma_omega_high), multi_sample=False if local else None)
            tr.sample('omega', alan.Normal(self.omega_mu, self.log_omega_sigma.exp()))

            #reading level
            sigma_obs_low = t.max(self.low, self.sigma_obs_low.exp())
            sigma_obs_high = t.min(self.high, self.sigma_obs_high.exp())
            tr.sample('sigma_obs', alan.Uniform(sigma_obs_low, sigma_obs_high))
            tr.sample('psi_int', alan.Normal(self.psi_int_mu, self.log_psi_int_sigma.exp()))

    covariates = {'basement': t.load('radon/data/basement.pt').rename('plate_state', 'plate_county', 'plate_zipcode', 'plate_reading',...).to(device),
        'county_uranium':t.load('radon/data/county_uranium.pt').rename('plate_state', 'plate_county').to(device)}

    data = {'obs':t.load('radon/data/radon.pt').rename('plate_state', 'plate_county', 'plate_zipcode', 'plate_reading',...).to(device)}


    return P, Q, data, covariates, {}, {}
