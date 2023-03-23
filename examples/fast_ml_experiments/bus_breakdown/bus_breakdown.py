import torch as t
import torch.nn as nn
import alan

def generate_model(N,M,device,ML=1):
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
    # data = {**covariates, **data}
    test_data = {'obs':t.load('bus_breakdown/data/delay_test.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).to(device)}
    # test_data = {**test_covariates, **test_data}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']],-1)}
    # all_data = {**all_covariates, **all_data}

    bus_company_name_dim = covariates['bus_company_name'].shape[-1]
    run_type_dim = covariates['run_type'].shape[-1]

    def P(tr, run_type, bus_company_name):
      '''
      Hierarchical Model
      '''

      #Year level

      tr('sigma_beta', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)))
      tr('mu_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))
      tr('beta', alan.Normal(tr['mu_beta'], tr['sigma_beta'].exp()), plates = 'plate_Year')

      #Borough level
      tr('sigma_alpha', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)), plates = 'plate_Borough')
      tr('alpha', alan.Normal(tr['beta'], tr['sigma_alpha'].exp()))

      #ID level
      tr('log_sigma_phi_psi', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)), plates = 'plate_ID')
      tr('psi', alan.Normal(t.zeros((run_type_dim,)).to(device), tr['log_sigma_phi_psi'].exp()), plates = 'plate_ID')
      tr('phi', alan.Normal(t.zeros((bus_company_name_dim,)).to(device), tr['log_sigma_phi_psi'].exp()), plates = 'plate_ID')
      tr('obs', alan.NegativeBinomial(total_count=130, logits=tr['alpha'] + tr['phi'] @ bus_company_name + tr['psi'] @ run_type))




    if ML == 1:
        class Q(alan.AlanModule):
            def __init__(self):
                super().__init__()
                #sigma_beta
                self.sigma_beta = alan.MLNormal()
                #mu_beta
                self.mu_beta = alan.MLNormal()
                #beta
                self.beta = alan.MLNormal({'plate_Year': M})
                #sigma_alpha
                self.sigma_alpha = alan.MLNormal({'plate_Borough': J})
                #alpha
                self.alpha = alan.MLNormal({'plate_Year': M,'plate_Borough': J})
                #log_sigma_phi_psi logits
                self.log_sigma_phi_psi = alan.MLNormal({'plate_ID':I})
                #psi
                self.psi = alan.MLNormal({'plate_ID':I}, sample_shape=(run_type_dim,))
                #phi
                self.phi = alan.MLNormal({'plate_ID':I}, sample_shape=(bus_company_name_dim,))


            def forward(self, tr, run_type, bus_company_name):
                #Year level

                tr('sigma_beta', self.sigma_beta())
                tr('mu_beta', self.mu_beta())
                tr('beta', self.beta())

                #Borough level
                tr('sigma_alpha', self.sigma_alpha())
                tr('alpha', self.alpha())

                #ID level
                tr('log_sigma_phi_psi', self.log_sigma_phi_psi())
                tr('psi', self.psi())
                tr('phi', self.phi())
    elif ML == 2:
        class Q(alan.AlanModule):
            def __init__(self):
                super().__init__()
                #sigma_beta
                self.sigma_beta = alan.ML2Normal()
                #mu_beta
                self.mu_beta = alan.ML2Normal()
                #beta
                self.beta = alan.ML2Normal({'plate_Year': M})
                #sigma_alpha
                self.sigma_alpha = alan.ML2Normal({'plate_Borough': J})
                #alpha
                self.alpha = alan.ML2Normal({'plate_Year': M,'plate_Borough': J})
                #log_sigma_phi_psi logits
                self.log_sigma_phi_psi = alan.ML2Normal({'plate_ID':I})
                #psi
                self.psi = alan.ML2Normal({'plate_ID':I}, sample_shape=(run_type_dim,))
                #phi
                self.phi = alan.ML2Normal({'plate_ID':I}, sample_shape=(bus_company_name_dim,))


            def forward(self, tr, run_type, bus_company_name):
                #Year level

                tr('sigma_beta', self.sigma_beta())
                tr('mu_beta', self.mu_beta())
                tr('beta', self.beta())

                #Borough level
                tr('sigma_alpha', self.sigma_alpha())
                tr('alpha', self.alpha())

                #ID level
                tr('log_sigma_phi_psi', self.log_sigma_phi_psi())
                tr('psi', self.psi())
                tr('phi', self.phi())

    return P, Q, data, covariates, test_data, test_covariates, all_data, all_covariates
