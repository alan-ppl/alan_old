import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
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
Ns = [10,30]
Ms = [10,50,100]



for N in Ns:
    for M in Ms:

        plate_1, plate_2 = dims(2 , [M,N])
        if N == 30:
            d_z = 20
        else:
            d_z = 5
        x = t.randn(M,N,d_z)[plate_1,plate_2].to(device)

        def P(tr):
          '''
          Heirarchical Model
          '''

          tr['mu_z'] = tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device))
          tr['psi_z'] = tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device))
          tr['psi_y'] = tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device))

          tr['z'] = tpp.Normal(tr['mu_z'] * t.ones((d_z)).to(device), tr['psi_z'].exp(), sample_dim=plate_1)

          tr['obs'] = tpp.Normal((tr['z'] @ x), tr['psi_y'].exp())


        data_y = tpp.sample(P,"obs")
        test_y

        t.save(data_y['obs'].order(*data_y['obs'].dims), 'data_y_{0}_{1}.pt'.format(N, M))
        t.save(x.order(*x.dims), 'weights_{0}_{1}.pt'.format(N,M))


        test_data_y = tpp.sample(P,"obs")
        t.save(data_y['obs'].order(*data_y['obs'].dims), 'test_data_y_{0}_{1}.pt'.format(N, M))
