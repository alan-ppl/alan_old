import torch as t
import alan
from alan.experiment_utils import seed_torch

def generate_model(N,M,device,ML=2, run=0, use_data=True):
    M = 3
    J = 3
    I = 30

    sizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':I}

    covariates = {'run_type': t.load('bus_breakdown/data/run_type_train_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
        'bus_company_name': t.load('bus_breakdown/data/bus_company_name_train_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    test_covariates = {'run_type': t.load('bus_breakdown/data/run_type_test_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
        'bus_company_name': t.load('bus_breakdown/data/bus_company_name_test_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    all_covariates = {'run_type': t.cat((covariates['run_type'],test_covariates['run_type']),2),
        'bus_company_name': t.cat([covariates['bus_company_name'],test_covariates['bus_company_name']],2)}

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
      tr('log_sigma_phi_psi', alan.Normal(tr.zeros(()), tr.ones(())))
      tr('psi', alan.Normal(tr.zeros((run_type_dim,)), tr['log_sigma_phi_psi'].exp()))
      tr('phi', alan.Normal(tr.zeros((bus_company_name_dim,)), tr['log_sigma_phi_psi'].exp()))
      # tr('theta', alan.Normal(np.log(1) * tr.ones(()), np.log(5) * tr.ones(())))
      # tr('obs', alan.NegativeBinomial(total_count=tr['theta'].exp(), logits=tr['alpha'] + tr['phi'] @ bus_company_name + tr['psi'] @ run_type))
      tr('obs', alan.Binomial(total_count=131, logits=tr['alpha'] + tr['phi'] @ bus_company_name + tr['psi'] @ run_type))




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
                self.log_sigma_phi_psi = alan.MLNormal()
                #psi
                self.psi = alan.MLNormal(sample_shape=(run_type_dim,))
                #phi
                self.phi = alan.MLNormal(sample_shape=(bus_company_name_dim,))
                #theta
                # self.theta = alan.MLNormal({'plate_ID':I})


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
                # tr('theta', self.theta())
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
                self.log_sigma_phi_psi = alan.ML2Normal()
                #psi
                self.psi = alan.ML2Normal(sample_shape=(run_type_dim,))
                #phi
                self.phi = alan.ML2Normal(sample_shape=(bus_company_name_dim,))
                #theta
                # self.theta = alan.MLNormal({'plate_ID':I})


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




    if use_data:
        data = {'obs':t.load('bus_breakdown/data/delay_train_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
        # print(data)
        test_data = {'obs':t.load('bus_breakdown/data/delay_test_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
        all_data = {'obs': t.cat([data['obs'],test_data['obs']],-1)}
    else:
        model = alan.Model(P)

        all_data = model.sample_prior(inputs = all_covariates)
        #data_prior_test = model.sample_prior(platesizes = sizes, inputs = test_covariates)
        data = all_data
        test_data = {}
        data['obs'], test_data['obs'] = t.split(all_data['obs'].clone(), [I,I], -1)

        all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1)}

    return P, Q, data, covariates, all_data, all_covariates, sizes

if __name__ == "__main__":
    for lr in [0.5]:
        print('ML1')
        seed_torch(0)
        P, Q, data, covariates, all_data, all_covariates, sizes = generate_model(2,2, t.device("cpu"), ML=1, run=0, use_data=False)

        
        model = alan.Model(P, Q())
        data = {'obs':data.pop('obs')}
        K = 10

        sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))

        scales = [model.Q.sigma_alpha.mean2conv(*model.Q.sigma_alpha.named_means)['scale']]
        grads_loc = {k:[] for k in sample.weights().keys()}
        grads_scale = {k:[] for k in sample.weights().keys()}
        for j in range(1):

            sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))
            elbo = sample.elbo()
            model.update(lr, sample)

            scales = [model.Q.sigma_alpha.mean2conv(*model.Q.sigma_alpha.named_means)['scale']]

            for k,v in sample.weights().items():
                grads_loc[k].append(model.Q.__getattr__(k).grad[0])
                grads_scale[k].append(model.Q.__getattr__(k).grad[1])

        print('ML2')
        seed_torch(0)
        P, Q, data, covariates, all_data, all_covariates, sizes = generate_model(2,2, t.device("cpu"), ML=2, run=0, use_data=False)


        
        model = alan.Model(P, Q())
        data = {'obs':data.pop('obs')}

        sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))

        scales = [model.Q.sigma_alpha.mean2conv(*model.Q.sigma_alpha.named_means)['scale']]
        grads2_loc = {k:[] for k in sample.weights().keys()}
        grads2_scale = {k:[] for k in sample.weights().keys()}
        for j in range(1):

            sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))
            elbo = sample.elbo()
            model.update(lr, sample)

            scales = [model.Q.sigma_alpha.mean2conv(*model.Q.sigma_alpha.named_means)['scale']]

            for k,v in sample.weights().items():
                grads2_loc[k].append(model.Q.__getattr__(k).grad[0])
                grads2_scale[k].append(model.Q.__getattr__(k).grad[1])



    for k in sample.weights().keys():
        if k == 'sigma_alpha':
            print(f'Scale grad difference for {k}: {(grads_scale[k][0] - grads2_scale[k][0]).abs()}')
            print(grads_scale[k][0])
            print(grads2_scale[k][0])
        
    