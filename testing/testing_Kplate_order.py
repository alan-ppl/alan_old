import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
import tqdm
from functorch.dim import dims

import numpy as np
import random

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)

seed_torch(0)
a = t.zeros(5,)
plate_1,plate_2, plate_3 = dims(3 , [2,3,4])
def P(tr):
  '''
  Bayesian Heirarchical Gaussian Model
  '''
  tr['mu'] = tpp.Normal(t.zeros(1,), t.ones(1,))

  tr['omega'] = tpp.Normal(t.zeros(1,), t.ones(1,))
  tr['psi'] = tpp.Normal(tr['mu'], t.ones(1,), sample_dim=plate_1)

  tr['phi'] = tpp.Normal(tr['psi'], t.ones(1,), sample_dim=plate_2)


  tr['obs'] = tpp.Normal(tr['phi'], tr['omega'].exp(), sample_dim=plate_3)



class Q(tpp.Q_module):
    def __init__(self):
        super().__init__()
        self.reg_param("m_mu", t.zeros((1,)))
        self.reg_param("log_s_mu", t.zeros((1,)))

        self.reg_param("m_omega", t.zeros((1,)))
        self.reg_param("log_s_omega", t.zeros((1,)))

        self.reg_param("m_psi", t.zeros((2,)), [plate_1])
        self.reg_param('log_s_psi', t.zeros((2,)), [plate_1])


        self.reg_param('m_phi', t.zeros((2,3)), [plate_1,plate_2])
        self.reg_param('log_s_phi', t.zeros((2,3)), [plate_1,plate_2])


    def forward(self, tr):
        tr['mu'] = tpp.Normal(self.m_mu, self.log_s_mu.exp())
        tr['psi'] = tpp.Normal(self.m_psi, self.log_s_psi.exp())
        tr['phi'] = tpp.Normal(self.m_phi, self.log_s_phi.exp())
        tr['omega'] = tpp.Normal(self.m_omega, self.log_s_omega.exp())

data = {'obs': t.tensor([[[[ 2.2161],
          [-1.1900],
          [ 1.8575]],

         [[-0.4844],
          [ 0.0783],
          [-1.4624]]],


        [[[ 2.7805],
          [ 0.0267],
          [ 1.9694]],

         [[-2.1758],
          [ 0.3924],
          [-1.0304]]],


        [[[ 1.3663],
          [ 0.4321],
          [ 2.7907]],

         [[-1.6035],
          [-0.5806],
          [-1.5620]]],


        [[[ 0.7598],
          [ 0.9892],
          [ 1.7875]],

         [[-1.2252],
          [ 0.7817],
          [ 0.9214]]]])[plate_3, plate_1, plate_2]}


model = tpp.Model(P, Q(), data)
print(data)
opt = t.optim.Adam(model.parameters(), lr=1E-3)

K= 5
print("K={}".format(K))
for i in range(1):
    opt.zero_grad()
    elbo = model.elbo(K=K)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())
