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

print('...', flush=True)


device = t.device("cuda" if t.cuda.is_available() else "cpu")

results_dict = {}

Ks = [5,10,15]
Ns = [10,30]
Ms = [10,50,100]



for K,M,N in itertools.product(Ks,Ms,Ns):
    print(K,M,N)
    results_dict[N] = results_dict.get(N, {})
    results_dict[N][M] = results_dict[N].get(M, {})
    results_dict[N][M][K] = results_dict[N][M].get(K, {})
    elbos = []
    for i in range(5):
        t.manual_seed(0)
        plate_1, plate_2 = dims(2 , [M,N])
        if N == 30:
            d_z = 20
        else:
            d_z = 5
        x = t.randn(M,N,d_z)[plate_1,plate_2].to(device)

        t.manual_seed(i)


        def P(tr):
          '''
          Heirarchical Model
          '''

          tr['mu_z'] = tpp.Normal(t.zeros((1,)).to(device), t.ones((1,)).to(device), sample_K=False)
          tr['psi_z'] = tpp.Normal(t.zeros((1,)).to(device), t.ones((1,)).to(device), sample_K=False)
          tr['psi_y'] = tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device), sample_K=False)

          tr['z'] = tpp.Normal(tr['mu_z'] * t.ones((d_z)).to(device), tr['psi_z'].exp() * t.ones((d_z)).to(device), sample_dim=plate_1)

          tr['obs'] = tpp.Normal((tr['z'] @ x), tr['psi_y'].exp())



        class Q(tpp.Q_module):
            def __init__(self):
                super().__init__()
                #mu_z
                self.reg_param("m_mu_z", t.zeros((1,)))
                self.reg_param("log_theta_mu_z", t.zeros((1,)))
                #psi_z
                self.reg_param("m_psi_z", t.zeros((1,)))
                self.reg_param("log_theta_psi_z", t.zeros((1,)))
                #psi_y
                self.reg_param("m_psi_y", t.zeros((1,)))
                self.reg_param("log_theta_psi_y", t.zeros((1,)))

                #z
                self.reg_param("mu", t.zeros((M,d_z)), [plate_1])
                self.reg_param("log_sigma", t.zeros((M, d_z)), [plate_1])


            def forward(self, tr):
                tr['mu_z'] = tpp.Normal(self.m_mu_z, self.log_theta_mu_z.exp(), sample_K=False)
                tr['psi_z'] = tpp.Normal(self.m_psi_z, self.log_theta_psi_z.exp(), sample_K=False)
                tr['psi_y'] = tpp.Normal(self.m_psi_y, self.log_theta_psi_y.exp(), sample_K=False)

                # sigma_z = self.sigma @ self.sigma.mT
                # eye = t.eye(d_z).to(device)
                # z_eye = eye * 0.001
                # sigma_z = sigma_z + z_eye
                #print(self.mu * t.ones((M,)).to(device)[plate_1])

                tr['z'] = tpp.Normal(self.mu, self.log_sigma.exp())





        data_y = tpp.sample(P,"obs")

        model = tpp.Model(P, Q(), data_y)
        model.to(device)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)


        dim = tpp.make_dims(P, K, [plate_1], exclude=['mu_z', 'psi_z', 'psi_y'])

        for i in range(50000):
            opt.zero_grad()
            elbo = model.elbo(dims=dim)
            (-elbo).backward()
            opt.step()

            if 0 == i%1000:
                print("Iteration: {0}, ELBO: {1:.2f}".format(i,elbo.item()))

        elbos.append(elbo.item())
    results_dict[N][M][K] = {'lower_bound':np.mean(elbos),'std':np.std(elbos)}

file = 'results_local_IW.json'
with open(file, 'w') as f:
    json.dump(results_dict, f)