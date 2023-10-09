import torch as t
import alan
from alan.experiment_utils import seed_torch
from alan.utils import reduce_Ks

def generate_model(N,M,device,ML=2, run=0, use_data=True, first='sigma'):
    # M = 3
    # J = 3
    # I = 30

    # sizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':I}

    # covariates = {'run_type': t.randn(M,J,I,5).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
    #     'bus_company_name': t.randn(M,J,I,4).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    # test_covariates = {'run_type': t.randn(M,J,I,5).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
    #     'bus_company_name': t.randn(M,J,I,4).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    # all_covariates = {'run_type': t.cat((covariates['run_type'],test_covariates['run_type']),2),
    #     'bus_company_name': t.cat([covariates['bus_company_name'],test_covariates['bus_company_name']],2)}

    I=2
    J=3
    M=4
    sizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':I}
    covariates = {'run_type': t.randn(M,J,I,5).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
        'bus_company_name': t.randn(M,J,I,4).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    test_covariates = {'run_type': t.randn(M,J,I,5).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
        'bus_company_name': t.randn(M,J,I,4).rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    all_covariates = {'run_type': t.cat((covariates['run_type'],test_covariates['run_type']),2),
        'bus_company_name': t.cat([covariates['bus_company_name'],test_covariates['bus_company_name']],2)}
    
    bus_company_name_dim = covariates['bus_company_name'].shape[-1]
    run_type_dim = covariates['run_type'].shape[-1]

    if first == 'sigma':
        def P(tr, run_type, bus_company_name):
            '''
            Hierarchical Model
            '''
            #Year level
            #Borough level
            tr('sigma_alpha', alan.Normal(1, 1), plates = ('plate_Borough'))
            tr('beta', alan.Normal(1, 1), plates = 'plate_Year')
            
            
            tr('alpha', alan.Normal(tr['beta'], tr['sigma_alpha'].exp()))

            # #ID level
            tr('obs', alan.Normal(tr['alpha'],1), plates = 'plate_ID')
            
    elif first == 'beta':
        def P(tr, run_type, bus_company_name):
            '''
            Hierarchical Model
            '''

            #Year level
            #Borough level
            tr('beta', alan.Normal(1, 1), plates = 'plate_Year')
            tr('sigma_alpha', alan.Normal(1, 1), plates = ('plate_Borough'))
            
            tr('alpha', alan.Normal(tr['beta'], tr['sigma_alpha'].exp()))

            # #ID level
            tr('obs', alan.Normal(tr['alpha'],1), plates = 'plate_ID')

    if ML == 1:
        class Q(alan.AlanModule):
            def __init__(self):
                super().__init__()

                # #beta
                self.beta = alan.MLNormal({'plate_Year': M})
                #sigma_alpha
                self.sigma_alpha = alan.MLNormal({'plate_Borough': J})
                #alpha
                self.alpha = alan.MLNormal({'plate_Borough': J,'plate_Year': M})


            def forward(self, tr, run_type, bus_company_name):
                #Year level
                tr('beta', self.beta())

                #Borough level
                tr('sigma_alpha', self.sigma_alpha())
                tr('alpha', self.alpha())

    elif ML == 2:
        class Q(alan.AlanModule):
            def __init__(self):
                super().__init__()
                # #beta
                self.beta = alan.ML2Normal({'plate_Year': M})
                #sigma_alpha
                self.sigma_alpha = alan.ML2Normal({'plate_Borough': J})
                #alpha
                self.alpha = alan.ML2Normal({'plate_Borough': J,'plate_Year': M})

            def forward(self, tr, run_type, bus_company_name):
                #Year level
                tr('beta', self.beta())

                #Borough level
                tr('sigma_alpha', self.sigma_alpha())
                tr('alpha', self.alpha())




    if use_data:
        # data = {'obs':t.load('bus_breakdown/data/delay_train_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
        # # print(data)
        # test_data = {'obs':t.load('bus_breakdown/data/delay_test_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
        # all_data = {'obs': t.cat([data['obs'],test_data['obs']],-1)}
        None
    else:
        model = alan.Model(P)

        all_data = model.sample_prior(inputs = all_covariates)
        #data_prior_test = model.sample_prior(platesizes = sizes, inputs = test_covariates)
        data = all_data
        test_data = {}

        data['obs'], test_data['obs'] = t.split(all_data['obs'].clone(), [I,I], 0)

        all_data = {'obs': t.cat([data['obs'],test_data['obs']], 0)}

    return P, Q, data, covariates, all_data, all_covariates, sizes

if __name__ == "__main__": 
    for first in ['sigma', 'beta']:
        lr = 0.5
        print(first + ' first')
        seed_torch(0)
        P, Q, data, covariates, all_data, all_covariates, sizes = generate_model(2,2, t.device("cpu"), ML=1, run=0, use_data=False, first=first)

        
        model = alan.Model(P, Q())
        data = {'obs':data.pop('obs')}
        K = 20

        T=10
        sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))

        scales = [model.Q.sigma_alpha.mean2conv(*model.Q.sigma_alpha.named_means)['scale']]
        grads_loc = {k:[] for k in sample.samples.keys()}
        grads_scale = {k:[] for k in sample.samples.keys()}
        elbos_1 = []
        for j in range(T):

            sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))
            elbo = sample.elbo()
            model.update(lr, sample)

            elbos_1.append(elbo)
            scales = [model.Q.sigma_alpha.mean2conv(*model.Q.sigma_alpha.named_means)['scale']]

            for k in sample.samples.keys():
                grads_loc[k].append(model.Q.__getattr__(k).grad[0])
                grads_scale[k].append(model.Q.__getattr__(k).grad[1])

        seed_torch(0)
        P, Q, data, covariates, all_data, all_covariates, sizes = generate_model(2,2, t.device("cpu"), ML=2, run=0, use_data=False)


        
        model = alan.Model(P, Q())
        data = {'obs':data.pop('obs')}

        sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))

        scales = [model.Q.sigma_alpha.mean2conv(*model.Q.sigma_alpha.named_means)['scale']]
        grads2_loc = {k:[] for k in sample.samples.keys()}
        grads2_scale = {k:[] for k in sample.samples.keys()}
        elf2 = {k:[] for k in sample.samples.keys()}
        elbos_2 = []
        for j in range(T):

            sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))
            elbo = sample.elbo()
            model.update(lr, sample)

            elbos_2.append(elbo)
            scales = [model.Q.sigma_alpha.mean2conv(*model.Q.sigma_alpha.named_means)['scale']]

            for k in sample.samples.keys():
                grads2_loc[k].append(model.Q.__getattr__(k).grad[0])
                grads2_scale[k].append(model.Q.__getattr__(k).grad[1])
                elf2[k].append(model.Q.__getattr__(k).elfs[1])
                
                
        print(f'Elbo ML1: {elbos_1[-1]}')
        print(f'Elbo ML2: {elbos_2[-1]}')
        

        for k in sample.samples.keys():
            print(f'Scale grad difference for {k}: {(grads_scale[k][-1] - grads2_scale[k][-1]).abs()}')







