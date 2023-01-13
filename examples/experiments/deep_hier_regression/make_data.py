import torch as t
import torch.nn as nn
import alan

from alan.experiment_utils import seed_torch


seed_torch(0)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

results_dict = {}

Ns = [4]
Ms = [2,4,10]



for N in Ns:
    for M in Ms:

        sizes = {'plate_muz2':2, 'plate_muz3':2, 'plate_muz4':2, 'plate_z':M, 'plate_obs':N}
        # plate_muz2, plate_muz3, plate_muz4, plate_z, plate_obs = dims(5 , [2,2,2,M,N])
        if N == 30:
            d_z = 20
        else:
            d_z = 5


        def P(tr):
          '''
          Heirarchical Model
          '''

          tr.sample('mu_z1', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)))
          tr.sample('mu_z2', alan.Normal(tr['mu_z1'], t.ones(()).to(device)), plates='plate_muz2')
          tr.sample('mu_z3', alan.Normal(tr['mu_z2'], t.ones(()).to(device)), plates='plate_muz3')
          tr.sample('mu_z4', alan.Normal(tr['mu_z3'], t.ones(()).to(device)), plates='plate_muz4')
          tr.sample('psi_y', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)))
          tr.sample('psi_z', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)))

          tr.sample('z', alan.Normal(tr['mu_z4'] * t.ones((d_z)).to(device), tr['psi_z'].exp()), plates='plate_z')


          tr.sample('obs', alan.Normal((tr['z'] @ tr['x']), tr['psi_y'].exp()))


        x = {'x': t.randn(2, 2, 2, M,N,d_z).rename('plate_muz2', 'plate_muz3', 'plate_muz4', 'plate_z', 'plate_obs', ...).to(device)}

        data_y = alan.sample(P, sizes, varnames=('obs',), covariates=x)

        x_test = {'x': t.randn(2, 2, 2, M,N//2,d_z).rename('plate_muz2', 'plate_muz3', 'plate_muz4', 'plate_z', 'plate_obs', ...).to(device)}

        test_data_y = alan.sample(P, {'plate_muz2':2, 'plate_muz3':2, 'plate_muz4':2, 'plate_z':M, 'plate_obs':N//2}, varnames=('obs',), covariates=x_test)


        t.save(x['x'].rename(None), 'data/weights_{0}_{1}.pt'.format(N,M))
        t.save(data_y['obs'].rename(None), 'data/data_y_{0}_{1}.pt'.format(N, M))
        t.save(x_test['x'].rename(None), 'data/test_weights_{0}_{1}.pt'.format(N,M))
        t.save(test_data_y['obs'].rename(None), 'data/test_data_y_{0}_{1}.pt'.format(N, M))
