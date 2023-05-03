import torch as t
import torch.nn as nn
import alan

def generate_model(M, J, I, device, local=False, AlanModule=False, run=0):
    
    # sizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':I}

    # covariates = {'run_type': t.load('data/run_type_train.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float().to(device),
    #     'bus_company_name': t.load('data/bus_company_name_train.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float().to(device)}
    # test_covariates = {'run_type': t.load('data/run_type_test.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float().to(device),
    #     'bus_company_name': t.load('data/bus_company_name_test.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float().to(device)}
    # all_covariates = {'run_type': t.cat([covariates['run_type'],test_covariates['run_type']],-2),
    #     'bus_company_name': t.cat([covariates['bus_company_name'],test_covariates['bus_company_name']],-2)}

    # data = {'obs':t.load('data/delay_train.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).to(device)}
    # test_data = {'obs':t.load('data/delay_test.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).to(device)}
    # all_data = {'obs': t.cat([data['obs'],test_data['obs']],-1)}

    # bus_company_name_dim = covariates['bus_company_name'].shape[-1]
    # run_type_dim = covariates['run_type'].shape[-1]

    sizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':I}

    covariates = {'run_type': t.load('data/run_type_train_{}.pt'.format(run+10)).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
        'bus_company_name': t.load('data/bus_company_name_train_{}.pt'.format(run+10)).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    test_covariates = {'run_type': t.load('data/run_type_test_{}.pt'.format(run+10)).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
        'bus_company_name': t.load('data/bus_company_name_test_{}.pt'.format(run+10)).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    all_covariates = {'run_type': t.cat([covariates['run_type'],test_covariates['run_type']],-3),
        'bus_company_name': t.cat([covariates['bus_company_name'],test_covariates['bus_company_name']],-3)}

    data = {'obs':t.load('data/delay_train_{}.pt'.format(run+10)).rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
    # data = {**covariates, **data}
    test_data = {'obs':t.load('data/delay_test_{}.pt'.format(run+10)).rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
    # test_data = {**test_covariates, **test_data}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']],-2)}
    # all_data = {**all_covariates, **all_data}

    bus_company_name_dim = covariates['bus_company_name'].shape[-1]
    run_type_dim = covariates['run_type'].shape[-1]

    def P(tr, run_type, bus_company_name):
        '''
        Hierarchical Model
        '''
        #Year level
        tr('sigma_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))#, plates = 'plate_Year') # YearVariance
        tr('mu_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))#, plates = 'plate_Year')    # YearMean
        tr('beta', alan.Normal(tr['mu_beta'], tr['sigma_beta'].exp()), plates='plate_Year')                          # BoroughMean

        #Borough level
        tr('sigma_alpha', alan.Normal(t.zeros(()).to(device), 0.25*t.ones(()).to(device)), plates = 'plate_Borough') # BoroughVariance
        tr('alpha', alan.Normal(tr['beta'], tr['sigma_alpha'].exp()))                                                # IDMean

        #ID level
        tr('log_sigma_phi_psi', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)), plates = 'plate_ID')          # WeightVariance
        tr('psi', alan.Normal(t.zeros((run_type_dim,)).to(device), tr['log_sigma_phi_psi'].exp()), plates = 'plate_ID')          # J
        tr('phi', alan.Normal(t.zeros((bus_company_name_dim,)).to(device), tr['log_sigma_phi_psi'].exp()), plates = 'plate_ID')  # C
        print(tr['phi'].dtype, bus_company_name.float().dtype)  # horrific
        if tr['phi'].dtype == t.float32:
            tr('obs', alan.NegativeBinomial(total_count=130, logits=tr['alpha'] + tr['phi'] @ bus_company_name.float() + tr['psi'] @ run_type.float()))  # Delay
        else:
            tr('obs', alan.NegativeBinomial(total_count=130, logits=tr['alpha'] + tr['phi'] @ bus_company_name + tr['psi'] @ run_type))  # Delay

    if not AlanModule:
        def Q(tr):
            '''
            Hierarchical Model
            '''
            #Year level
            tr('sigma_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))#, plates = 'plate_Year') # YearVariance
            tr('mu_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))#, plates = 'plate_Year')    # YearMean
            tr('beta', alan.Normal(tr['mu_beta'], tr['sigma_beta'].exp()), plates='plate_Year')                          # BoroughMean

            #Borough level
            tr('sigma_alpha', alan.Normal(t.zeros(()).to(device), 0.25*t.ones(()).to(device)), plates = 'plate_Borough') # BoroughVariance
            tr('alpha', alan.Normal(tr['beta'], tr['sigma_alpha'].exp()))                                                # IDMean

            #ID level
            tr('log_sigma_phi_psi', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)), plates = 'plate_ID')          # WeightVariance
            tr('psi', alan.Normal(t.zeros((run_type_dim,)).to(device), tr['log_sigma_phi_psi'].exp()), plates = 'plate_ID')          # J
            tr('phi', alan.Normal(t.zeros((bus_company_name_dim,)).to(device), tr['log_sigma_phi_psi'].exp()), plates = 'plate_ID')  # C
            # tr('obs', alan.NegativeBinomial(total_count=130, logits=tr['alpha'] + tr['phi'] @ tr['bus_company_name'] + tr['psi'] @ tr['run_type']))  # Delay
        
    else:
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

                # self.high = t.tensor(10.0).to(device)
                # self.low = t.tensor(0.0).to(device)

            def forward(self, tr, run_type, bus_company_name):
                #Year level

                tr('sigma_beta', alan.Normal(self.sigma_beta_mean, 0.0001*self.sigma_beta_sigma.exp()))
                tr('mu_beta', alan.Normal(self.mu_beta_mean, 0.0001*self.log_mu_beta_sigma.exp()))
                tr('beta', alan.Normal(self.beta_mu, self.log_beta_sigma.exp()))

                #Borough level
                tr('sigma_alpha', alan.Normal(self.sigma_alpha_mean, 0.25*self.sigma_alpha_sigma.exp()))
                tr('alpha', alan.Normal(self.alpha_mu, self.log_alpha_sigma.exp()))

                #ID level
                tr('log_sigma_phi_psi', alan.Normal(self.log_sigma_phi_psi_mean, 0.0001*self.log_sigma_phi_psi_sigma.exp()))
                tr('psi', alan.Normal(self.psi_mean, self.log_psi_sigma.exp()))
                tr('phi', alan.Normal(self.phi_mean, self.log_phi_sigma.exp()))


    return P, Q, data, covariates, all_data, all_covariates