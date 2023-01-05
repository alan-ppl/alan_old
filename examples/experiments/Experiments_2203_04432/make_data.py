import torch as t
import torch.nn as nn
import alan

from alan.experiment_utils import seed_torch


seed_torch(0)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

results_dict = {}

Ks = [1,3,10,30]
Ns = [10,30]
Ms = [10,50,100]



for N in Ns:
    for M in Ms:

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


        x = {'x': t.randn(M,N,d_z).rename('plate_1', 'plate_2', ...).to(device)}

        data_y = alan.sample(P, sizes, varnames=('obs',), covariates=x)

        x_test = {'x': t.randn(M,N,d_z).rename('plate_1', 'plate_2', ...).to(device)}

        test_data_y = alan.sample(P, sizes, varnames=('obs',), covariates=x_test)


        t.save(x['x'].rename(None), 'data/weights_{0}_{1}.pt'.format(N,M))
        t.save(data_y['obs'].rename(None), 'data/data_y_{0}_{1}.pt'.format(N, M))
        t.save(x_test['x'].rename(None), 'data/test_weights_{0}_{1}.pt'.format(N,M))
        t.save(test_data_y['obs'].rename(None), 'data/test_data_y_{0}_{1}.pt'.format(N, M))
