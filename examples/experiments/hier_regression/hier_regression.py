import torch as t
import torch.nn as nn
import alan

def generate_model(N,M,local,device):
    sizes = {'plate_1':M, 'plate_2':N}
    if N == 30:
        d_z = 20
    else:
        d_z = 5

    def P(tr):
      '''
      Heirarchical Model
      '''

      tr.sample('mu_z', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)))
      tr.sample('psi_z', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)))
      tr.sample('psi_y', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)))

      tr.sample('z', alan.Normal(tr['mu_z'] * t.ones((d_z)).to(device), tr['psi_z'].exp()), plates='plate_1')

      tr.sample('obs', alan.Normal((tr['z'] @ tr['x']), tr['psi_y'].exp()))




    class Q(alan.QModule):
        def __init__(self):
            super().__init__()
            #mu_z
            self.m_mu_z = nn.Parameter(t.zeros(()))
            self.log_theta_mu_z = nn.Parameter(t.zeros(()))
            #psi_z
            self.m_psi_z = nn.Parameter(t.zeros(()))
            self.log_theta_psi_z = nn.Parameter(t.zeros(()))
            #psi_y
            self.m_psi_y = nn.Parameter(t.zeros(()))
            self.log_theta_psi_y = nn.Parameter(t.zeros(()))


            #z
            self.mu = nn.Parameter(t.zeros((M,d_z), names=('plate_1',None)))
            self.log_sigma = nn.Parameter(t.zeros((M,d_z), names=('plate_1',None)))


        def forward(self, tr):
            tr.sample('mu_z', alan.Normal(self.m_mu_z, self.log_theta_mu_z.exp()), multi_sample=False if local else None)
            tr.sample('psi_z', alan.Normal(self.m_psi_z, self.log_theta_psi_z.exp()), multi_sample=False if local else None)
            tr.sample('psi_y', alan.Normal(self.m_psi_y, self.log_theta_psi_y.exp()), multi_sample=False if local else None)


            tr.sample('z', alan.Normal(self.mu, self.log_sigma.exp()))

    covariates = {'x':t.load('hier_regression/data/weights_{0}_{1}.pt'.format(N,M)).rename('plate_1','plate_2',...).to(device)}
    test_covariates = {'x':t.load('hier_regression/data/test_weights_{0}_{1}.pt'.format(N,M)).rename('plate_1','plate_2',...).to(device)}
    all_covariates = {'x': t.vstack([covariates['x'],test_covariates['x']])}

    data = {'obs':t.load('hier_regression/data/data_y_{0}_{1}.pt'.format(N, M)).rename('plate_1','plate_2').to(device)}
    test_data = {'obs':t.load('hier_regression/data/test_data_y_{0}_{1}.pt'.format(N, M)).rename('plate_1','plate_2').to(device)}
    all_data = {'obs': t.vstack([data['obs'],test_data['obs']])}

    return P, Q, data, covariates, all_data, all_covariates
