import torch as t
import torch.nn as nn
import alan
from alan.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from alan.backend import vi
import tqdm
from functorch.dim import dims
import argparse
import json
import numpy as np
import itertools

t.manual_seed(0)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

results_dict = {}

Ks = [1,5,10,15]
Ns = [2]
Ms = [2,4,10]



for N in Ns:
    for M in Ms:

        plate_muz2, plate_muz3, plate_muz4, plate_z, plate_obs = dims(5 , [2,2,2,M,N])
        if N == 30:
            d_z = 20
        else:
            d_z = 5
        x = t.randn(2, 2, 2, M,N,d_z)[plate_muz2, plate_muz3, plate_muz4, plate_z, plate_obs].to(device)

        def P(tr):
          '''
          Heirarchical Model
          '''

          tr['mu_z1'] = alan.Normal(t.zeros(()).to(device), t.ones(()).to(device))
          tr['mu_z2'] = alan.Normal(tr['mu_z1'], t.ones(()).to(device), sample_dim=plate_muz2)
          tr['mu_z3'] = alan.Normal(tr['mu_z2'], t.ones(()).to(device), sample_dim=plate_muz3)
          tr['mu_z4'] = alan.Normal(tr['mu_z3'], t.ones(()).to(device), sample_dim=plate_muz4)
          tr['psi_z'] = alan.Normal(t.zeros(()).to(device), t.ones(()).to(device))
          tr['psi_y'] = alan.Normal(t.zeros(()).to(device), t.ones(()).to(device))

          tr['z'] = alan.Normal(tr['mu_z4'] * t.ones((d_z)).to(device), tr['psi_z'].exp(), sample_dim=plate_z)

          tr['obs'] = alan.Normal((tr['z'] @ x), tr['psi_y'].exp())



        data_y = alan.sample(P,"obs")
        test_data_y = alan.sample(P,"obs")
        t.save(data_y['obs'].order(*data_y['obs'].dims), 'data_y_{0}_{1}.pt'.format(N, M))
        t.save(x.order(*x.dims), 'weights_{0}_{1}.pt'.format(N,M))
        t.save(test_data_y['obs'].order(*data_y['obs'].dims), 'test_data_y_{0}_{1}.pt'.format(N, M))
