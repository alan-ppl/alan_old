import torch as t
import torch.nn as nn
import alan

def generate_model(N,M,device, ML=1):
    sizes = {'plate_1':M, 'plate_2':N}
    d_z = 18
    def P(tr, x):
      '''
      Heirarchical Model
      '''

      tr('mu_z', alan.Normal(tr.zeros((d_z,)).to(device), tr.ones((d_z,)).to(device)))
      tr('psi_z', alan.Normal(tr.zeros((d_z,)).to(device), tr.ones((d_z,)).to(device)))

      tr('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()), plates='plate_1')

      tr('obs', alan.Bernoulli(logits = tr['z'] @ x))


    class Q(alan.AlanModule):
        def __init__(self):
            super().__init__()
            #mu_z
            self.m_mu_z = nn.Parameter(t.zeros((d_z,)))
            self.log_theta_mu_z = nn.Parameter(t.zeros((d_z,)))
            #psi_z
            self.m_psi_z = nn.Parameter(t.zeros((d_z,)))
            self.log_theta_psi_z = nn.Parameter(t.zeros((d_z,)))

            #z
            self.mu = nn.Parameter(t.zeros((M,d_z), names=('plate_1',None)))
            self.log_sigma = nn.Parameter(t.zeros((M,d_z), names=('plate_1',None)))


        def forward(self, tr, x):
            tr('mu_z', alan.Normal(self.m_mu_z, self.log_theta_mu_z.exp()))
            tr('psi_z', alan.Normal(self.m_psi_z, self.log_theta_psi_z.exp()))

            tr('z', alan.Normal(self.mu, self.log_sigma.exp()))



    covariates = {'x':t.load('movielens/data/weights_{0}_{1}.pt'.format(N,M)).to(device)}
    test_covariates = {'x':t.load('movielens/data/test_weights_{0}_{1}.pt'.format(N,M)).to(device)}
    all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
    covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)
    test_covariates['x'] = test_covariates['x'].rename('plate_1','plate_2',...)

    data = {'obs':t.load('movielens/data/data_y_{0}_{1}.pt'.format(N, M)).to(device)}
    test_data = {'obs':t.load('movielens/data/test_data_y_{0}_{1}.pt'.format(N, M)).to(device)}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
    data['obs'] = data['obs'].rename('plate_1','plate_2')
    test_data['obs'] = test_data['obs'].rename('plate_1','plate_2')
    return P, Q, data, covariates, test_data, test_covariates, all_data, all_covariates
