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

          tr['mu_z1'] = tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device))
          tr['mu_z2'] = tpp.Normal(tr['mu_z1'], t.ones(()).to(device), sample_dim=plate_muz2)
          tr['mu_z3'] = tpp.Normal(tr['mu_z2'], t.ones(()).to(device), sample_dim=plate_muz3)
          tr['mu_z4'] = tpp.Normal(tr['mu_z3'], t.ones(()).to(device), sample_dim=plate_muz4)
          tr['psi_z'] = tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device))
          tr['psi_y'] = tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device))

          tr['z'] = tpp.Normal(tr['mu_z4'] * t.ones((d_z)).to(device), tr['psi_z'].exp(), sample_dim=plate_z)

          tr['obs'] = tpp.Normal((tr['z'] @ x), tr['psi_y'].exp())



        class Q(tpp.Q_module):
            def __init__(self):
                super().__init__()
                #mu_z1
                self.reg_param("m_mu_z1", t.zeros(()))
                self.reg_param("log_theta_mu_z1", t.zeros(()))
                #mu_z2
                self.reg_param("m_mu_z2", t.zeros((2,)), [plate_muz2])
                self.reg_param("log_theta_mu_z2", t.zeros((2,)), [plate_muz2])
                #mu_z3
                self.reg_param("m_mu_z3", t.zeros((2,2)), [plate_muz2, plate_muz3])
                self.reg_param("log_theta_mu_z3", t.zeros((2,2)), [plate_muz2, plate_muz3])
                #mu_z4
                self.reg_param("m_mu_z4", t.zeros((2,2,2)), [plate_muz2, plate_muz3, plate_muz4])
                self.reg_param("log_theta_mu_z4", t.zeros((2,2,2)), [plate_muz2, plate_muz3, plate_muz4])
                #psi_z
                self.reg_param("m_psi_z", t.zeros(()))
                self.reg_param("log_theta_psi_z", t.zeros(()))
                #psi_y
                self.reg_param("m_psi_y", t.zeros(()))
                self.reg_param("log_theta_psi_y", t.zeros(()))

                #z
                self.reg_param("mu", t.zeros((2, 2, 2, M, d_z)), [plate_muz2, plate_muz3, plate_muz4, plate_1])
                self.reg_param("log_sigma", t.zeros((2, 2, 2, M, d_z)), [plate_muz2, plate_muz3, plate_muz4, plate_1])


            def forward(self, tr):
                tr['mu_z1'] = tpp.Normal(self.m_mu_z1, self.log_theta_mu_z1.exp())
                tr['mu_z2'] = tpp.Normal(self.m_mu_z2, self.log_theta_mu_z2.exp())
                tr['mu_z3'] = tpp.Normal(self.m_mu_z3, self.log_theta_mu_z3.exp())
                tr['mu_z4'] = tpp.Normal(self.m_mu_z4, self.log_theta_mu_z4.exp())
                tr['psi_z'] = tpp.Normal(self.m_psi_z, self.log_theta_psi_z.exp())
                tr['psi_y'] = tpp.Normal(self.m_psi_y, self.log_theta_psi_y.exp())


                tr['z'] = tpp.Normal(self.mu, self.log_sigma.exp())

        data_y = tpp.sample(P,"obs")

        print(data_y['obs'].dims)
        t.save(data_y['obs'].order(*data_y['obs'].dims), 'data_y_{0}_{1}.pt'.format(N, M))
        t.save(x.order(*x.dims), 'weights_{0}_{1}.pt'.format(N,M))
