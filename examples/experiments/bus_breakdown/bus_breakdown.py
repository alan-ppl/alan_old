import torch as t
import torch.nn as nn
import alan

def generate_model(N,M,local,device):
    M = 3
    J = 3
    I = 30

    sizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':I}

    covariates = {'run_type': t.load('bus_breakdown/data/run_type_train.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float().to(device),
        'bus_company_name': t.load('bus_breakdown/data/bus_company_name_train.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float().to(device)}
    test_covariates = {'run_type': t.load('bus_breakdown/data/run_type_test.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float().to(device),
        'bus_company_name': t.load('bus_breakdown/data/bus_company_name_test.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float().to(device)}
    all_covariates = {'run_type': t.cat([covariates['run_type'],test_covariates['run_type']],-2),
        'bus_company_name': t.cat([covariates['bus_company_name'],test_covariates['bus_company_name']],-2)}

    data = {'obs':t.load('bus_breakdown/data/delay_train.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).to(device)}
    test_data = {'obs':t.load('bus_breakdown/data/delay_test.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).to(device)}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']],-1)}

    bus_company_name_dim = covariates['bus_company_name'].shape[-1]
    run_type_dim = covariates['run_type'].shape[-1]

    def P(tr):
      '''
      Hierarchical Model
      '''

      #Year level
      tr.sample('sigma_beta', alan.Categorical(t.tensor([0.1,0.5,0.4,0.05,0.05]).to(device)))
      tr.sample('mu_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))
      tr.sample('beta', alan.Normal(tr['mu_beta'], tr['sigma_beta'].exp()), plates = 'plate_Year')

      #Borough level
      tr.sample('sigma_alpha', alan.Categorical(t.tensor([0.1,0.4,0.05,0.5,0.05]).to(device)), plates = 'plate_Borough')
      tr.sample('alpha', alan.Normal(tr['beta'], tr['sigma_alpha'].exp()))

      #ID level
      tr.sample('log_sigma_phi_psi', alan.Categorical(t.tensor([0.1,0.4,0.5,0.05,0.05]).to(device)), plates = 'plate_ID')
      tr.sample('psi', alan.Normal(t.zeros((run_type_dim,)).to(device), tr['log_sigma_phi_psi'].exp()), plates = 'plate_ID')
      tr.sample('phi', alan.Normal(t.zeros((bus_company_name_dim,)).to(device), tr['log_sigma_phi_psi'].exp()), plates = 'plate_ID')
      tr.sample('obs', alan.NegativeBinomial(total_count=130, logits=tr['alpha'] + tr['phi'] @ tr['bus_company_name'] + tr['psi'] @ tr['run_type']))





    class Q(alan.QModule):
        def __init__(self):
            super().__init__()
            #sigma_beta
            self.sigma_beta_logits = nn.Parameter(t.randn((5,)))
            #mu_beta
            self.mu_beta_mean = nn.Parameter(t.zeros(()))
            self.log_mu_beta_sigma = nn.Parameter(t.zeros(()))
            #beta
            self.beta_mu = nn.Parameter(t.zeros((M,),names=('plate_Year',)))
            self.log_beta_sigma = nn.Parameter(t.zeros((M,), names=('plate_Year',)))
            #sigma_alpha
            self.sigma_alpha_logits = nn.Parameter(t.randn((J,5), names=('plate_Borough',None)))
            #alpha
            self.alpha_mu = nn.Parameter(t.zeros((M,J), names=('plate_Year', 'plate_Borough')))
            self.log_alpha_sigma = nn.Parameter(t.zeros((M,J), names=('plate_Year', 'plate_Borough')))
            #log_sigma_phi_psi logits
            self.log_sigma_phi_psi_logits = nn.Parameter(t.randn((I,5), names=('plate_ID',None)))
            #psi
            self.psi_mean = nn.Parameter(t.zeros((I,run_type_dim), names=('plate_ID',None)))
            self.log_psi_sigma = nn.Parameter(t.zeros((I,run_type_dim), names=('plate_ID',None)))
            #phi
            self.phi_mean = nn.Parameter(t.zeros((I,bus_company_name_dim), names=('plate_ID',None)))
            self.log_phi_sigma = nn.Parameter(t.zeros((I,bus_company_name_dim), names=('plate_ID',None)))

            self.high = t.tensor(10.0).to(device)
            self.low = t.tensor(0.0).to(device)

        def forward(self, tr):
            #Year level

            tr.sample('sigma_beta', alan.Categorical(logits=self.sigma_beta_logits), multi_sample=False if local else True)
            tr.sample('mu_beta', alan.Normal(self.mu_beta_mean, self.log_mu_beta_sigma.exp()), multi_sample=False if local else True)
            tr.sample('beta', alan.Normal(self.beta_mu, self.log_beta_sigma.exp()), multi_sample=False if local else True)

            #Borough level
            tr.sample('sigma_alpha', alan.Categorical(logits=self.sigma_alpha_logits), multi_sample=False if local else True)
            tr.sample('alpha', alan.Normal(self.alpha_mu, self.log_alpha_sigma.exp()), multi_sample=False if local else True)

            #ID level
            tr.sample('log_sigma_phi_psi', alan.Categorical(logits=self.log_sigma_phi_psi_logits))
            tr.sample('psi', alan.Normal(self.psi_mean, self.log_psi_sigma.exp()))
            tr.sample('phi', alan.Normal(self.phi_mean, self.log_phi_sigma.exp()))


    return P, Q, data, covariates, all_data, all_covariates
