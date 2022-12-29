import torch as t
import torch.nn as nn
import alan
from alan.prob_prog import Trace, TraceLogP, TraceSampleLogQ
import tqdm
from functorch.dim import dims
import argparse
import json
import numpy as np
import itertools
import random

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)

seed_torch(0)
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

          tr['mu_z'] = alan.Normal(t.zeros(()).to(device), t.ones(()).to(device))
          tr['psi_z'] = alan.Normal(t.zeros(()).to(device), t.ones(()).to(device))
          tr['psi_y'] = alan.Normal(t.zeros(()).to(device), t.ones(()).to(device))

          tr['z'] = alan.Normal(tr['mu_z'] * t.ones((d_z)).to(device), tr['psi_z'].exp(), sample_dim=plate_1)

          tr['obs'] = alan.Normal((tr['z'] @ x), tr['psi_y'].exp())


        data_y = alan.sample(P,"obs")

        t.save(data_y['obs'].order(*data_y['obs'].dims), 'data_y_{0}_{1}.pt'.format(N, M))
        t.save(x.order(*x.dims), 'weights_{0}_{1}.pt'.format(N,M))
