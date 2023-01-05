import torch as t
import torch.nn as nn
import alan

def generate_model(N,M,local,device):

    sizes = {'plate_muz2':2, 'plate_muz3':2, 'plate_muz4':2, 'plate_z':M, 'plate_obs':N}
    if N == 30:
        d_z = 20
    else:
        d_z = 5

    def P(tr):

      tr.sample('mu_z1', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)), group='local' if local else None)
      tr.sample('mu_z2', alan.Normal(tr['mu_z1'], t.ones(()).to(device)), plates='plate_muz2', group='local' if local else None)
      tr.sample('mu_z3', alan.Normal(tr['mu_z2'], t.ones(()).to(device)), plates='plate_muz3', group='local' if local else None)
      tr.sample('mu_z4', alan.Normal(tr['mu_z3'], t.ones(()).to(device)), plates='plate_muz4', group='local' if local else None)
      tr.sample('psi_y', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)), group='local' if local else None)
      tr.sample('psi_z', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)), group='local' if local else None)

      tr.sample('z', alan.Normal(tr['mu_z4'] * t.ones((d_z)).to(device), tr['psi_z'].exp()), plates='plate_z')


      tr.sample('obs', alan.Normal((tr['z'] @ tr['x']), tr['psi_y'].exp()))



    class Q(alan.QModule):
        def __init__(self):
            super().__init__()
            #mu_z1
            self.m_mu_z1 = nn.Parameter(t.zeros(()))
            self.log_theta_mu_z1 = nn.Parameter(t.zeros(()))
            #mu_z2
            self.m_mu_z2 = nn.Parameter(t.zeros((2,), names=('plate_muz2',)))
            self.log_theta_mu_z2 = nn.Parameter(t.zeros((2,), names=('plate_muz2',)))
            #mu_z3
            self.m_mu_z3 = nn.Parameter(t.zeros((2,2), names=('plate_muz2', 'plate_muz3')))
            self.log_theta_mu_z3 = nn.Parameter(t.zeros((2,2), names=('plate_muz2', 'plate_muz3')))
            #mu_z4
            self.m_mu_z4 = nn.Parameter(t.zeros((2,2,2), names=('plate_muz2', 'plate_muz3', 'plate_muz4')))
            self.log_theta_mu_z4 = nn.Parameter(t.zeros((2,2,2), names=('plate_muz2', 'plate_muz3', 'plate_muz4')))
            #psi_z
            self.m_psi_z = nn.Parameter(t.zeros(()))
            self.log_theta_psi_z = nn.Parameter(t.zeros(()))
            #psi_y
            self.m_psi_y = nn.Parameter(t.zeros(()))
            self.log_theta_psi_y = nn.Parameter(t.zeros(()))

            #z
            self.mu = nn.Parameter(t.zeros((2, 2, 2, M,d_z), names = ('plate_muz2', 'plate_muz3', 'plate_muz4', 'plate_z', None)))
            self.log_sigma = nn.Parameter(t.zeros((2, 2, 2, M,d_z), names = ('plate_muz2', 'plate_muz3', 'plate_muz4', 'plate_z', None)))


        def forward(self, tr):
            tr.sample('mu_z1', alan.Normal(self.m_mu_z1, self.log_theta_mu_z1.exp()))
            tr.sample('mu_z2', alan.Normal(self.m_mu_z2, self.log_theta_mu_z2.exp()))
            tr.sample('mu_z3', alan.Normal(self.m_mu_z3, self.log_theta_mu_z3.exp()))
            tr.sample('mu_z4', alan.Normal(self.m_mu_z4, self.log_theta_mu_z4.exp()))
            tr.sample('psi_z', alan.Normal(self.m_psi_z, self.log_theta_psi_z.exp()))
            tr.sample('psi_y', alan.Normal(self.m_psi_y, self.log_theta_psi_y.exp()))


            tr.sample('z', alan.Normal(self.mu, self.log_sigma.exp()))

    covariates = {'x':t.load('data/weights_{0}_{1}.pt'.format(N,M)).rename('plate_muz2', 'plate_muz3', 'plate_muz4', 'plate_z', 'plate_obs', ...).to(device)}
    test_covariates = {'x':t.load('data/test_weights_{0}_{1}.pt'.format(N,M)).rename('plate_muz2', 'plate_muz3', 'plate_muz4', 'plate_z', 'plate_obs', ...).to(device)}
    all_covariates = {'x': t.vstack([covariates['x'],test_covariates['x']])}

    data = {'obs':t.load('data/data_y_{0}_{1}.pt'.format(N, M)).rename('plate_muz2', 'plate_muz3', 'plate_muz4','plate_obs', 'plate_z').to(device)}
    test_data = {'obs':t.load('data/test_data_y_{0}_{1}.pt'.format(N, M)).rename('plate_muz2', 'plate_muz3', 'plate_muz4', 'plate_obs', 'plate_z').to(device)}
    all_data = {'obs': t.vstack([data['obs'],test_data['obs']])}

    return P, Q, data, covariates, all_data, all_covariates
