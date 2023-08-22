import torch as t
import alan
from alan.experiment_utils import seed_torch

def generate_model(N,M, run=0, use_data=True):
    M = 2
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
    #   tr('obs', alan.Binomial(total_count=131, logits=tr['alpha'] + tr['phi'] @ bus_company_name + tr['psi'] @ run_type))
      tr('obs', alan.Bernoulli(logits=tr['alpha'] + tr['phi'] @ bus_company_name + tr['psi'] @ run_type))




    class Q(alan.AlanModule):
        def __init__(self):
            super().__init__()
            #sigma_beta
            self.sigma_beta = alan.AMMP_ISNormal()
            #mu_beta
            self.mu_beta = alan.AMMP_ISNormal()
            #beta
            self.beta = alan.AMMP_ISNormal({'plate_Year': M})
            #sigma_alpha
            self.sigma_alpha = alan.AMMP_ISNormal({'plate_Borough': J})
            #alpha
            self.alpha = alan.AMMP_ISNormal({'plate_Year': M,'plate_Borough': J})
            #log_sigma_phi_psi logits
            self.log_sigma_phi_psi = alan.AMMP_ISNormal()
            #psi
            self.psi = alan.AMMP_ISNormal(sample_shape=(run_type_dim,))
            #phi
            self.phi = alan.AMMP_ISNormal(sample_shape=(bus_company_name_dim,))
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
    seed_torch(0)
    P, Q, data, covariates, all_data, all_covariates, sizes = generate_model(2,2, run=0, use_data=False)


    model = alan.Model(P, Q())
    data = {'obs':data.pop('obs')}
    K = 10

    for j in range(2000):

        sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))
        elbo = sample.elbo()
        model.ammpis_update(0.3, sample)




        for i in range(2):
            try:
                sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))
                pred_likelihood = model.predictive_ll(sample, N = 10, data_all=all_data, inputs_all=all_covariates)
                break
            except:
                pred_likelihood = 0

        if j % 10 == 0:
            print(f'Elbo: {elbo.item()}')
            print(f'Pred_ll: {pred_likelihood}')
