import torch as t
import torch.nn as nn
import alan

def generate_model(N,M,device, ML=1, run=0):
    sizes = {'plate_1':M, 'plate_2':N}
    d_z = 18
    def P(tr, x):
      '''
      Heirarchical Model
      '''

      tr('mu_z', alan.Normal(tr.zeros((d_z,)), tr.ones((d_z,))))
      tr('psi_z', alan.Normal(tr.zeros((d_z,)), tr.ones((d_z,))))

      tr('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()), plates='plate_1')

      tr('obs', alan.Bernoulli(logits = tr['z'] @ x))


    if ML == 1:
        class Q(alan.AlanModule):
            def __init__(self):
                super().__init__()
                #mu_z
                self.mu = alan.MLNormal(sample_shape=(d_z,))
                #psi_z
                self.psi_z = alan.MLNormal(sample_shape=(d_z,))

                #z
                self.z = alan.MLNormal({'plate_1': M},sample_shape=(d_z,))


            def forward(self, tr,x):
                tr('mu_z', self.mu())
                tr('psi_z', self.psi_z())

                tr('z', self.z())

    elif ML == 2:
        class Q(alan.AlanModule):
            def __init__(self):
                super().__init__()
                #mu_z
                self.mu = alan.ML2Normal(sample_shape=(d_z,))
                #psi_z
                self.psi_z = alan.ML2Normal(sample_shape=(d_z,))

                #z
                self.z = alan.ML2Normal({'plate_1': M},sample_shape=(d_z,))


            def forward(self, tr,x):
                tr('mu_z', self.mu())
                tr('psi_z', self.psi_z())

                tr('z', self.z())



    covariates = {'x':t.load('movielens/data/weights_{0}_{1}_{2}.pt'.format(N, M,run)).to(device)}
    test_covariates = {'x':t.load('movielens/data/test_weights_{0}_{1}_{2}.pt'.format(N, M,run)).to(device)}
    all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
    covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)
    test_covariates['x'] = test_covariates['x'].rename('plate_1','plate_2',...)


    data = {'obs':t.load('movielens/data/data_y_{0}_{1}_{2}.pt'.format(N, M,run)).to(device)}
    test_data = {'obs':t.load('movielens/data/test_data_y_{0}_{1}_{2}.pt'.format(N, M,run)).to(device)}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
    data['obs'] = data['obs'].rename('plate_1','plate_2')
    test_data['obs'] = test_data['obs'].rename('plate_1','plate_2')
    return P, Q, data, covariates, test_data, test_covariates, all_data, all_covariates
