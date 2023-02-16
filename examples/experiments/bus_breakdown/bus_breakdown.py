import torch as t
import torch.nn as nn
import alan

def generate_model(M, J, I, device, local=False):
    
    sizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':I}

    covariates = {'run_type': t.load('data/run_type_train.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float().to(device),
        'bus_company_name': t.load('data/bus_company_name_train.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float().to(device)}
    test_covariates = {'run_type': t.load('data/run_type_test.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float().to(device),
        'bus_company_name': t.load('data/bus_company_name_test.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float().to(device)}
    all_covariates = {'run_type': t.cat([covariates['run_type'],test_covariates['run_type']],-2),
        'bus_company_name': t.cat([covariates['bus_company_name'],test_covariates['bus_company_name']],-2)}

    data = {'obs':t.load('data/delay_train.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).to(device)}
    test_data = {'obs':t.load('data/delay_test.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).to(device)}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']],-1)}

    bus_company_name_dim = covariates['bus_company_name'].shape[-1]
    run_type_dim = covariates['run_type'].shape[-1]

    def P(tr):
        '''
        Hierarchical Model
        '''
        #Year level
        tr.sample('sigma_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))#, plates = 'plate_Year') # YearVariance
        tr.sample('mu_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))#, plates = 'plate_Year')    # YearMean
        tr.sample('beta', alan.Normal(tr['mu_beta'], tr['sigma_beta'].exp()), plates='plate_Year')                          # BoroughMean

        #Borough level
        tr.sample('sigma_alpha', alan.Normal(t.zeros(()).to(device), 0.25*t.ones(()).to(device)), plates = 'plate_Borough') # BoroughVariance
        tr.sample('alpha', alan.Normal(tr['beta'], tr['sigma_alpha'].exp()))                                                # IDMean

        #ID level
        tr.sample('log_sigma_phi_psi', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)), plates = 'plate_ID')          # WeightVariance
        tr.sample('psi', alan.Normal(t.zeros((run_type_dim,)).to(device), tr['log_sigma_phi_psi'].exp()), plates = 'plate_ID')          # J
        tr.sample('phi', alan.Normal(t.zeros((bus_company_name_dim,)).to(device), tr['log_sigma_phi_psi'].exp()), plates = 'plate_ID')  # C
        tr.sample('obs', alan.NegativeBinomial(total_count=130, logits=tr['alpha'] + tr['phi'] @ tr['bus_company_name'] + tr['psi'] @ tr['run_type']))  # Delay

    def Q(tr):
        '''
        Hierarchical Model
        '''
        #Year level
        tr.sample('sigma_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))#, plates = 'plate_Year') # YearVariance
        tr.sample('mu_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))#, plates = 'plate_Year')    # YearMean
        tr.sample('beta', alan.Normal(tr['mu_beta'], tr['sigma_beta'].exp()), plates='plate_Year')                          # BoroughMean

        #Borough level
        tr.sample('sigma_alpha', alan.Normal(t.zeros(()).to(device), 0.25*t.ones(()).to(device)), plates = 'plate_Borough') # BoroughVariance
        tr.sample('alpha', alan.Normal(tr['beta'], tr['sigma_alpha'].exp()))                                                # IDMean

        #ID level
        tr.sample('log_sigma_phi_psi', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)), plates = 'plate_ID')          # WeightVariance
        tr.sample('psi', alan.Normal(t.zeros((run_type_dim,)).to(device), tr['log_sigma_phi_psi'].exp()), plates = 'plate_ID')          # J
        tr.sample('phi', alan.Normal(t.zeros((bus_company_name_dim,)).to(device), tr['log_sigma_phi_psi'].exp()), plates = 'plate_ID')  # C
        # tr.sample('obs', alan.NegativeBinomial(total_count=130, logits=tr['alpha'] + tr['phi'] @ tr['bus_company_name'] + tr['psi'] @ tr['run_type']))  # Delay

    return P, Q, data, covariates, all_data, all_covariates
