import torch as t
import torch.nn as nn
import alan
from alan.postproc import *
t.manual_seed(0)
import time

from alan.experiment_utils import seed_torch, n_mean

from torch.distributions import Normal, Uniform
from ammpis import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-N', type=int, default=5)
parser.add_argument('-K', type=int, default=10)
parser.add_argument('-T', type=int, default=100)
parser.add_argument('-b', '--backup_plots',  default=False, action='store_true')
parser.add_argument('-s', '--seed', type=int, default=0)
args = parser.parse_args()

N = args.N
K = args.K
T = args.T
backup_plots = args.backup_plots
seed = args.seed

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
colours = ['#9e9ac8','#fbb4b9','#253494','#de2d26','#31a354', '#fed976', '#feb24c']

fig_iters, ax_iters = plt.subplots(1,1, figsize=(8.0, 5.0))

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Using device: {device}")

seed_torch(seed)

dim_latent = N
# K=3
# T=100

num_runs = 10
lr = 0.01

prior_mean = Normal(0,150).sample((dim_latent,1)).float().to(device)
prior_scale = Uniform(1, 2).sample((dim_latent,1)).float().to(device)
lik_scale = Uniform(1, 2).sample((dim_latent,)).float().to(device)


def P(tr):
    tr('mu', alan.Normal(prior_mean.squeeze(1), prior_scale.squeeze(1)))
    tr('obs', alan.Normal(tr['mu'], lik_scale))

class Q_ml1(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.mu = alan.MLNormal(sample_shape=(dim_latent,))

    def forward(self, tr):
        tr('mu', self.mu())

class Q_ml2(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.mu = alan.ML2Normal(sample_shape=(dim_latent,))

    def forward(self, tr):
        tr('mu', self.mu())


def P_separate(tr):
    tr('mu1', alan.Normal(prior_mean[0], prior_scale[0]))
    tr('mu2', alan.Normal(prior_mean[1], prior_scale[1]))
    tr('mu3', alan.Normal(prior_mean[2], prior_scale[2]))
    tr('mu4', alan.Normal(prior_mean[3], prior_scale[3]))
    tr('mu5', alan.Normal(prior_mean[4], prior_scale[4]))

    mu = t.stack([tr['mu1'], tr['mu2'], tr['mu3'], tr['mu4'], tr['mu5']]).to(device)

    tr('obs', alan.Normal(mu.squeeze(1), lik_scale))

class Q_ml_separate(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.mu1 = alan.MLNormal(sample_shape=(1,))
        self.mu2 = alan.MLNormal(sample_shape=(1,))
        self.mu3 = alan.MLNormal(sample_shape=(1,))
        self.mu4 = alan.MLNormal(sample_shape=(1,))
        self.mu5 = alan.MLNormal(sample_shape=(1,))

    def forward(self, tr):
        tr('mu1', self.mu1())
        tr('mu2', self.mu2())
        tr('mu3', self.mu3())
        tr('mu4', self.mu4())
        tr('mu5', self.mu5())

class Q_ml2_separate(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.mu1 = alan.ML2Normal(sample_shape=(1,))
        self.mu2 = alan.ML2Normal(sample_shape=(1,))
        self.mu3 = alan.ML2Normal(sample_shape=(1,))
        self.mu4 = alan.ML2Normal(sample_shape=(1,))
        self.mu5 = alan.ML2Normal(sample_shape=(1,))

    def forward(self, tr):
        tr('mu1', self.mu1())
        tr('mu2', self.mu2())
        tr('mu3', self.mu3())
        tr('mu4', self.mu4())
        tr('mu5', self.mu5())

#Posterior
data = alan.Model(P).to(device).sample_prior(varnames='obs')

prior_params = t.cat([prior_mean, prior_scale], dim=1)
lik_params = lik_scale
init = t.tensor([0.0,1.0], dtype=t.float64).repeat((dim_latent,1)).to(device)

times_all = {'natural_rws': t.zeros((T+1,)),
             'ml1_toy': t.zeros((T+1,)),
             'ml2_toy': t.zeros((T+1,)),
             'ml1': t.zeros((T+1,)),
             'ml2': t.zeros((T+1,)),
             'ml1_separated': t.zeros((T+1,)), 
             'ml2_separated': t.zeros((T+1,))}

for i in range(num_runs):
    print(f"Run {i+1}/{num_runs}")

    seed_torch(seed)
    m_q, l_one_iters, entropies, times = natural_rws(T, init, lr, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None), device=device)
    times_all['natural_rws'] += times
    print("Natural RWS done.")

    seed_torch(seed)
    m_q_ml1, l_one_iters_ml1, entropies, times = ml1(T, init, lr, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None), device=device)
    times_all['ml1_toy'] += times
    print("ML1 Toy done.")

    seed_torch(seed)
    m_q_ml2, l_one_iters_ml2, entropies, times = ml2(T, init, lr, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None), device=device)
    times_all['ml2_toy'] += times
    print("ML2 Toy done.")

    seed_torch(seed)
    q = Q_ml1()
    m1 = alan.Model(P, q).condition(data=data)
    m1.to(device)

    elbos_ml1 = []
    times = t.zeros((T+1,))
    start = time.time()
    for i in range(T):
        sample = m1.sample_same(K, reparam=False, device=device)
        elbos_ml1.append(sample.elbo().item()) 
        m1.update(lr, sample)
        times[i+1] = time.time() - start
    times_all['ml1'] += times
    print("ML1_Regular done.")

    seed_torch(seed)
    q = Q_ml_separate()
    m1 = alan.Model(P_separate, q).condition(data=data)#.to(device)
    m1.to(device)

    elbos_ml1 = []
    times = t.zeros((T+1,))
    start = time.time()
    for i in range(T):
        sample = m1.sample_same(K, reparam=False, device=device)
        elbos_ml1.append(sample.elbo().item()) 
        m1.update(lr, sample)
        times[i+1] = time.time() - start
    times_all['ml1_separated'] += times

    print("ML1_Separated done.")

    seed_torch(seed)
    q = Q_ml2()
    m2 = alan.Model(P, q).condition(data=data)#.to(device)
    m2.to(device)

    elbos_ml2 = []
    times = t.zeros((T+1,))
    start = time.time()
    for i in range(T):
        sample = m2.sample_same(K, reparam=False, device=device)
        elbos_ml2.append(sample.elbo().item()) 
        m2.update(lr, sample)
        times[i+1] = time.time() - start
    times_all['ml2'] += times
    print("ML2_Regular done.")

    seed_torch(seed)
    q = Q_ml2_separate()
    m2 = alan.Model(P_separate, q).condition(data=data)#.to(device)
    m2.to(device)

    elbos_ml2 = []
    times = t.zeros((T+1,))
    start = time.time()
    for i in range(T):
        sample = m2.sample_same(K, reparam=False, device=device)
        elbos_ml2.append(sample.elbo().item()) 
        m2.update(lr, sample)
        times[i+1] = time.time() - start
    times_all['ml2_separated'] += times
    print("ML2_Separated done.")

for key in times_all:
    times_all[key] /= num_runs

# Plot each method's times
colour_count = 0
for method, times in times_all.items():
    plt.plot(times, label=method, color=colours[colour_count])
    colour_count += 1

# Add labels and legend
plt.xlabel('Iteration')
plt.ylabel('Time (s)')
plt.legend()
plt.title(f"ML Speed Test (N={N}, K={K}, T={T})")

plt.savefig(f"figures/ml_speed_test{'' if device == 'cpu' else '_gpu'}.png")
