import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from functorch.dim import dims, Dim
import argparse
import json
import numpy as np
import itertools

t.manual_seed(0)
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


plate_1, plate_2 = dims(2 , [M,N])

x_train = t.load('data/weights_{0}_{1}.pt'.format(N,M))[plate_1,plate_2].to(device)
x_test = t.load('data/test_weights_{0}_{1}.pt'.format(N,M))[plate_1,plate_2].to(device)
d_z = 18
class P(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = x
    def forward(self, tr):
        '''
        Heirarchical Model
        '''

        tr['mu_z'] = tpp.Normal(t.zeros((2,d_z)).to(device), t.ones((2,d_z)).to(device))
        tr['psi_z'] = tpp.Categorical(t.tensor([0.1,0.5,0.4,0.05,0.05]).to(device))

        tr['z'] = tpp.Normal(tr['mu_z'], tr['psi_z'].exp(), sample_dim=plate_1)
        tr['obs'] = tpp.Bernoulli(logits = tr['z'] @ self.x)

class Q(tpp.Q_module):
    def __init__(self):
        super().__init__()
        #mu_z
        self.reg_param("m_mu_z", t.zeros((d_z,)))
        self.reg_param("log_theta_mu_z", t.zeros((d_z,)))
        #psi_z
        self.reg_param('psi_z_logits', t.randn(5))

        #z
        self.reg_param("mu", t.zeros((M,d_z)), [plate_1])
        self.reg_param("log_sigma", t.zeros((M, d_z)), [plate_1])


    def forward(self, tr):
        tr['mu_z'] = tpp.Normal(self.m_mu_z, self.log_theta_mu_z.exp())
        tr['psi_z'] = tpp.Categorical(logits=self.psi_z_logits)

        tr['z'] = tpp.Normal(self.mu, self.log_sigma.exp())





data_y = {'obs':t.load('data/data_y_{0}_{1}.pt'.format(N, M))[plate_1,plate_2].to(device)}
test_data_y = {'obs':t.load('data/test_data_y_{0}_{1}.pt'.format(N, M))[plate_1,plate_2].to(device)}
for K in Ks:
    print(K,M,N)
    results_dict[N] = results_dict.get(N, {})
    results_dict[N][M] = results_dict[N].get(M, {})
    results_dict[N][M][K] = results_dict[N][M].get(K, {})
    elbos = []
    pred_liks = []
    for i in range(5):

        t.manual_seed(i)

        model = tpp.Model(P(x_train), Q(), data_y)
        model.to(device)

        opt = t.optim.Adam(model.parameters(), lr=1E-4)


        dim = tpp.make_dims(model.P, K, [plate_1])

        for i in range(1000):
            opt.zero_grad()
            theta_loss, phi_loss = model.rws(dims=dim)
            (theta_loss + phi_loss).backward()
            opt.step()

            if 0 == i%1000:
                print("Iteration: {0}, ELBO: {1:.2f}".format(i,phi_loss.item()))

        test_model = tpp.Model(P(x_test), model.Q, test_data_y)
        dim = tpp.make_dims(model.P, 1)
        pred_likelihood = test_model.pred_likelihood(dims=dim, test_data=test_data_y, num_samples=1000, reparam=False)
        pred_liks.append(pred_likelihood.item())
    results_dict[N][M][K] = {'lower_bound':np.mean(elbos),'std':np.std(elbos), 'elbos': elbos, 'pred_mean':np.mean(pred_liks), 'pred_std':np.std(pred_liks), 'preds':pred_liks}

file = 'results/movielens_results_discrete_variance_rws_N{0}_M{1}.json'.format(N,M)
with open(file, 'w') as f:
    json.dump(results_dict, f)
