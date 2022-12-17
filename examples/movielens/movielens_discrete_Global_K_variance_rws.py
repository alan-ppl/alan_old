import torch as t
import torch.nn as nn
import tpp

import argparse
import json
import numpy as np
import itertools
import time
import random

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)

seed_torch(0)
parser = argparse.ArgumentParser(description='Run the Heirarchical regression task.')

parser.add_argument('N', type=int,
                    help='Scale of experiment')
parser.add_argument('M', type=int,
                    help='Number of groups')

args = parser.parse_args()

print('...', flush=True)


device = t.device("cuda" if t.cuda.is_available() else "cpu")

results_dict = {}

Ks = [5,10,15]


np.random.seed(0)

M = args.M
N = args.N


sizes = {'plate_1':M, 'plate_2':N}
x_train = {'x': t.load('data/weights_{0}_{1}.pt'.format(N,M)).rename('plate_1','plate_2',...).to(device)}
x_test = {'x':t.load('data/test_weights_{0}_{1}.pt'.format(N,M)).rename('plate_1','plate_2',...).to(device)}
d_z = 18
class P(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = x
    def forward(self, tr):
        '''
        Heirarchical Model
        '''

        tr.sample('mu_z', tpp.Normal(t.zeros((d_z,)).to(device), t.ones((d_z,)).to(device)), group='group_1')
        tr.sample('psi_z', tpp.Categorical(t.tensor([0.1,0.5,0.4,0.05,0.05]).to(device)), group='group_1')

        tr.sample('z', tpp.Normal(tr['mu_z'], tr['psi_z'].exp()), plate='plate_1')
        tr.sample('obs', tpp.Bernoulli(logits = tr['z'] @ tr['x']))

class Q(tpp.Q):
    def __init__(self):
        super().__init__()
        #mu_z
        self.reg_param("m_mu_z", t.zeros((d_z,)))
        self.reg_param("log_theta_mu_z", t.zeros((d_z,)))
        #psi_z
        self.reg_param('psi_z_logits', t.randn(5))

        #z
        self.reg_param("mu", t.zeros((M,d_z)), ['plate_1'])
        self.reg_param("log_sigma", t.zeros((M, d_z)), ['plate_1'])


    def forward(self, tr):
        tr.sample('mu_z', tpp.Normal(self.m_mu_z, self.log_theta_mu_z.exp()))
        tr.sample('psi_z', tpp.Categorical(logits=self.psi_z_logits))

        tr.sample('z', tpp.Normal(self.mu, self.log_sigma.exp()))





data_y = {'obs':t.load('data/data_y_{0}_{1}.pt'.format(N, M)).rename('plate_1','plate_2').to(device)}
test_data_y = {'obs':t.load('data/test_data_y_{0}_{1}.pt'.format(N, M)).rename('plate_1','plate_2').to(device)}
for K in Ks:
    print(K,M,N)
    results_dict[N] = results_dict.get(N, {})
    results_dict[N][M] = results_dict[N].get(M, {})
    results_dict[N][M][K] = results_dict[N][M].get(K, {})
    elbos = []
    pred_liks = []
    times = []
    for i in range(5):
        seed_torch(i)
        start = time.time()

        model = tpp.Model(P(x_train), Q(), data_y | x_train)
        model.to(device)

        opt = t.optim.Adam(model.parameters(), lr=1E-4)



        for i in range(50000):
            opt.zero_grad()
            wake_theta_loss, wake_phi_loss = model.rws(K=K)
            (-wake_theta_loss + wake_phi_loss).backward()
            opt.step()

            if 0 == i%1000:
                print("Iteration: {0}, ELBO: {1:.2f}".format(i,wake_phi_loss.item()))

        times.append(time.time() - start)
        test_model = tpp.Model(P(x_test), model.Q, test_data_y | x_test)
        pred_likelihood = test_model.pred_likelihood(test_data=test_data_y, num_samples=1000, reparam=False)
        pred_liks.append(pred_likelihood.item())
    results_dict[N][M][K] = {'pred_mean':np.mean(pred_liks), 'pred_std':np.std(pred_liks), 'preds':pred_liks, 'avg_time':np.mean(times)}

file = 'results/movielens_results_global_K_rws_N{0}_M{1}.json'.format(N,M)
with open(file, 'w') as f:
    json.dump(results_dict, f)
