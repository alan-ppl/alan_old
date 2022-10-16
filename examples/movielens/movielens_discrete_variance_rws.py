import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from movielens_utils import get_ratings, get_features
from functorch.dim import dims
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

Ks = [1,5,10,15]
# Ns = [10,30]
# Ms = [10,50,100]

np.random.seed(0)

M = args.M
N = args.N

x = get_features()
users = np.random.choice(x.shape[0], M, replace=False)
films = np.random.choice(x.shape[1], N, replace=False)
plate_1, plate_2 = dims(2 , [M,N])

x = get_features()[np.ix_(users ,films)][plate_1,plate_2]
d_z = 18
def P(tr):
    '''
    Heirarchical Model
    '''

    tr['mu_z'] = tpp.Normal(t.zeros((d_z,)).to(device), t.ones((d_z,)).to(device))
    tr['psi_z'] = tpp.Categorical(t.tensor([0.1,0.5,0.4,0.05,0.05]))
    tr['z'] = tpp.Normal(tr['mu_z'], tr['psi_z'].exp(), sample_dim=plate_1)
    tr['obs'] = tpp.Bernoulli(logits = tr['z'] @ x)



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





data_y = {'obs':get_ratings()[np.ix_(users ,films)][plate_1,plate_2]}

for K in Ks:
    print(K,M,N)
    results_dict[N] = results_dict.get(N, {})
    results_dict[N][M] = results_dict[N].get(M, {})
    results_dict[N][M][K] = results_dict[N][M].get(K, {})
    elbos = []

    for i in range(5):

        t.manual_seed(i)

        model = tpp.Model(P, Q(), data_y)
        model.to(device)

        opt = t.optim.Adam(model.parameters(), lr=1E-4)


        dim = tpp.make_dims(P, K, [plate_1])

        for i in range(50000):
            opt.zero_grad()
            theta_loss, phi_loss = model.rws(dims=dim)
            (theta_loss + phi_loss).backward()
            opt.step()

            if 0 == i%1000:
                print("Iteration: {0}, ELBO: {1:.2f}".format(i,phi_loss.item()))

        elbos.append(phi_loss.item())
    results_dict[N][M][K] = {'lower_bound':np.mean(elbos),'std':np.std(elbos), 'elbos': elbos}

file = 'results/movielens_results_discrete_variance_rws_N{0}_M{1}.json'.format(N,M)
with open(file, 'w') as f:
    json.dump(results_dict, f)
