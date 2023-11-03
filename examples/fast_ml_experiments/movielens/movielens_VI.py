import torch as t
import torch.nn as nn
import alan
from alan.experiment_utils import seed_torch

def generate_model(N,M,device, ML=2, run=0, use_data=True, adjust_scale=False):
    sizes = {'plate_1':M, 'plate_2':N}
    d_z = 18
    if adjust_scale:
        def P(tr, x):
            '''
            Heirarchical Model
            '''

            tr('mu_z', alan.Normal(tr.zeros((d_z,)), 100*tr.ones((d_z,))))
            tr('psi_z', alan.Normal(tr.zeros((d_z,)), 0.01*tr.ones((d_z,))))

            tr('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()), plates='plate_1')

            tr('obs', alan.Bernoulli(logits = tr['z'] @ x))
    else:
        def P(tr, x):
            '''
            Heirarchical Model
            '''

            tr('mu_z', alan.Normal(tr.zeros((d_z,)), tr.ones((d_z,))))
            tr('psi_z', alan.Normal(tr.zeros((d_z,)), tr.ones((d_z,))))

            tr('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()), plates='plate_1')

            tr('obs', alan.Bernoulli(logits = tr['z'] @ x))

    class Q(alan.AlanModule):
        def __init__(self):
            super().__init__()
            #mu_z
            self.m_mu_z = nn.Parameter(t.zeros((d_z,)))
            self.log_mu_z_sigma = nn.Parameter(t.zeros((d_z,)))
            #psi_z
            self.m_psi_z = nn.Parameter(t.zeros((d_z,)))
            self.log_psi_z_sigma = nn.Parameter(t.zeros((d_z,)))

            #z
            self.mu = nn.Parameter(t.zeros((M,d_z), names=('plate_1',None)))
            self.log_z_sigma = nn.Parameter(t.zeros((M,d_z), names=('plate_1',None)))


        def forward(self, tr, x):
            tr('mu_z', alan.Normal(self.m_mu_z, self.log_mu_z_sigma.exp()))
            tr('psi_z', alan.Normal(self.m_psi_z, self.log_psi_z_sigma.exp()))

            tr('z', alan.Normal(self.mu, self.log_z_sigma.exp()))



    covariates = {'x':t.load('movielens/data/weights_{0}_{1}_{2}.pt'.format(N, M,run))}
    test_covariates = {'x':t.load('movielens/data/test_weights_{0}_{1}_{2}.pt'.format(N, M,run))}
    all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
    covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)
    test_covariates['x'] = test_covariates['x'].rename('plate_1','plate_2',...)

    if use_data:
        data = {'obs':t.load('movielens/data/data_y_{0}_{1}_{2}.pt'.format(N, M,run))}
        test_data = {'obs':t.load('movielens/data/test_data_y_{0}_{1}_{2}.pt'.format(N, M,run))}
        all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
        data['obs'] = data['obs'].rename('plate_1','plate_2')
        test_data['obs'] = test_data['obs'].rename('plate_1','plate_2')
    else:
        model = alan.Model(P, Q())
        all_data = model.sample_prior(platesizes = {'plate_1':300}, inputs = all_covariates)
        #data_prior_test = model.sample_prior(platesizes = sizes, inputs = test_covariates)
        data = all_data
        test_data = {}
        data['obs'], test_data['obs'] = t.split(all_data['obs'].clone(), [5,5], -1)
        all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1)}

    return P, Q, data, covariates, all_data, all_covariates, sizes


if __name__ == "__main__":
    seed_torch(0)
    P, Q, data, covariates, all_data, all_covariates, sizes = generate_model(5,300, t.device("cpu"), run=0, use_data=False)


    model = alan.Model(P, Q())
    data = {'obs':data.pop('obs')}
    K = 1
    opt = t.optim.Adam(model.parameters(), lr=0.1)
    for j in range(2000):
        opt.zero_grad()
        sample = model.sample_perm(K, data=data, inputs=covariates, reparam=True, device=t.device('cpu'))
        elbo = sample.elbo()
        (-elbo).backward()
        opt.step()



        for i in range(10):
            try:
                sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))
                pred_likelihood = model.predictive_ll(sample, N = 10, data_all=all_data, inputs_all=all_covariates)
                break
            except:
                pred_likelihood = 0

        if j % 100 == 0:
            print(f'Elbo: {elbo.item()}')
            print(f'Pred_ll: {pred_likelihood}')
