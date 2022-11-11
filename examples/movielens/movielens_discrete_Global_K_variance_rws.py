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
d_z = 18
def P_train(tr):
    '''
    Heirarchical Model
    '''

    tr['mu_z'] = tpp.Normal(t.zeros((2,d_z)).to(device), t.ones((2,d_z)).to(device))
    tr['psi_z'] = tpp.Normal(t.zeros((2,d_z)).to(device), t.ones((2,d_z)).to(device))
    tr['phi'] = tpp.Multinomial(1,t.tensor([0.1,0.9]))
    # print(tr['phi'])
    # print(tr['mu_z'])
    tr['z'] = tpp.Normal((tr['phi'] @ tr['mu_z']), tr['phi'] @ tr['psi_z'].exp(), sample_dim=plate_1)
    tr['obs'] = tpp.Bernoulli(logits = tr['z'] @ x_train)

x_test = t.load('data/test_weights_{0}_{1}.pt'.format(N,M))[plate_1,plate_2].to(device)
d_z = 18
def P_test(tr):
    '''
    Heirarchical Model
    '''

    tr['mu_z'] = tpp.Normal(t.zeros((2,d_z)).to(device), t.ones((2,d_z)).to(device))
    tr['psi_z'] = tpp.Normal(t.zeros((2,d_z)).to(device), t.ones((2,d_z)).to(device))
    tr['phi'] = tpp.Multinomial(1,t.tensor([0.1,0.9]))
    # print(tr['phi'])
    # print(tr['mu_z'])
    tr['z'] = tpp.Normal((tr['phi'] @ tr['mu_z']), tr['phi'] @ tr['psi_z'].exp(), sample_dim=plate_1)
    tr['obs'] = tpp.Bernoulli(logits = tr['z'] @ x_test)

class Q(tpp.Q_module):
    def __init__(self):
        super().__init__()
        #mu_z
        self.reg_param("m_mu_z", t.zeros((2,d_z)))
        self.reg_param("log_theta_mu_z", t.zeros((2,d_z)))
        #psi_z
        self.reg_param("m_psi_z", t.zeros((2,d_z)))
        self.reg_param("log_theta_psi_z", t.zeros((2,d_z)))
        #phi
        self.reg_param('prob', t.randn(2))
        #z
        self.reg_param("mu", t.zeros((M,d_z)), [plate_1])
        self.reg_param("log_sigma", t.zeros((M, d_z)), [plate_1])


    def forward(self, tr):
        tr['mu_z'] = tpp.Normal(self.m_mu_z, self.log_theta_mu_z.exp())
        tr['psi_z'] = tpp.Normal(self.m_psi_z, self.log_theta_psi_z.exp())
        tr['phi'] = tpp.Multinomial(1,logits = self.prob)
        # print(tr['phi'].shape)
        tr['z'] = tpp.Normal(self.mu, self.log_sigma.exp())





data_y = {'obs':t.load('data/data_y_{0}_{1}.pt'.format(N, M))[plate_1,plate_2].to(device)}
test_data_y = {'obs':t.load('data/test_data_y_{0}_{1}.pt'.format(N, M))[plate_1,plate_2].to(device)}
for K_size in Ks:
    print(K_size,M,N)
    results_dict[N] = results_dict.get(N, {})
    results_dict[N][M] = results_dict[N].get(M, {})
    results_dict[N][M][K_size] = results_dict[N][M].get(K_size, {})
    elbos = []
    pred_liks = []
    K_group1 = Dim(name='K_group1', size=K_size)
    K = Dim(name='K', size=K_size)
    dim = {'K':K, 'mu_z':K_group1, 'z':K_group1, 'psi_z': K_group1}
    for i in range(5):

        t.manual_seed(i)

        model = tpp.Model(P_train, Q(), data_y)
        model.to(device)

        opt = t.optim.Adam(model.parameters(), lr=1E-4)


        # dim = tpp.make_dims(P, K, [plate_1], exclude=['mu_z', 'psi_z'])
        # K_group1 = Dim(name='K_group1', size=K)
        # K = Dim(name='K', size=K)
        # dim = {'K':K, 'mu_z':K_group1, 'z':K_group1, 'psi_z': K_group1}
        for i in range(50000):
            opt.zero_grad()
            theta_loss, phi_loss = model.rws(dims=dim)
            (theta_loss + phi_loss).backward()
            opt.step()

            if 0 == i%1000:
                print("Iteration: {0}, ELBO: {1:.2f}".format(i,phi_loss.item()))

        test_model = tpp.Model(P_test, model.Q, test_data_y)
        dim = tpp.make_dims(P_test, 1)
        pred_likelihood = test_model.pred_likelihood(dims=dim, test_data=test_data_y, num_samples=1000, reparam=False).sum()
        pred_liks.append(pred_likelihood.item())
    results_dict[N][M][K] = {'lower_bound':np.mean(elbos),'std':np.std(elbos), 'elbos': elbos, 'pred_mean':np.mean(pred_liks), 'pred_std':np.std(pred_liks), 'preds':pred_liks}

file = 'results/movielens_results_global_K_rws_N{0}_M{1}.json'.format(N,M)
with open(file, 'w') as f:
    json.dump(results_dict, f)
