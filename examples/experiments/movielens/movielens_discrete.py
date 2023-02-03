import torch as t
import torch.nn as nn
import alan

def generate_model(N,M,local,device):
    sizes = {'plate_1':M, 'plate_2':N}
    d_z = 18
    def P(tr):
        '''
        Heirarchical Model
        '''

        tr.sample('mu_z', alan.Normal(t.zeros((d_z,)).to(device), t.ones((d_z,)).to(device)))
        tr.sample('psi_z', alan.Categorical(t.tensor([0.1,0.5,0.4,0.05,0.05]).to(device)))

        tr.sample('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()), plates='plate_1')
        tr.sample('obs', alan.Bernoulli(logits = tr['z'] @ tr['x']))

    class Q(alan.QModule):
        def __init__(self):
            super().__init__()
            #mu_z
            self.m_mu_z = nn.Parameter(t.zeros((d_z,)))
            self.log_theta_mu_z = nn.Parameter(t.zeros((d_z,)))
            #psi_z
            self.psi_z_logits = nn.Parameter(t.randn(5))

            #z
            self.mu = nn.Parameter(t.zeros((M,d_z), names=('plate_1',None)))
            self.log_sigma = nn.Parameter(t.zeros((M,d_z), names=('plate_1',None)))


        def forward(self, tr):
            tr.sample('mu_z', alan.Normal(self.m_mu_z, self.log_theta_mu_z.exp()), multi_sample=False if local else True)
            tr.sample('psi_z', alan.Categorical(logits=self.psi_z_logits), multi_sample=False if local else True)

            tr.sample('z', alan.Normal(self.mu, self.log_sigma.exp()))

    covariates = {'x':t.load('movielens/data/weights_{0}_{1}.pt'.format(N,M)).to(device)}
    test_covariates = {'x':t.load('movielens/data/test_weights_{0}_{1}.pt'.format(N,M)).to(device)}
    all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
    covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)


    data = {'obs':t.load('movielens/data/data_y_{0}_{1}.pt'.format(N, M)).to(device)}
    test_data = {'obs':t.load('movielens/data/test_data_y_{0}_{1}.pt'.format(N, M)).to(device)}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
    data['obs'] = data['obs'].rename('plate_1','plate_2')
    return P, Q, data, covariates, all_data, all_covariates
