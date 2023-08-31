import torch as t
import torch.nn as nn
import alan
from alan.postproc import *
t.manual_seed(0)

from alan.experiment_utils import seed_torch, n_mean

from torch.distributions import Normal, Uniform
from ammpis import *


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
colours = ['#9e9ac8','#fbb4b9','#253494','#de2d26','#31a354']

fig_iters, ax_iters = plt.subplots(1,1, figsize=(5.0, 5.0))

seed_torch(0)

num_latents = 200
K=100

T=500
prior_mean = Normal(0,150).sample((num_latents,1)).float()
prior_scale = Uniform(1, 2).sample((num_latents,1)).float()

lik_scale = Uniform(1, 2).sample((num_latents,)).float()
def P(tr):
  '''
  Bayesian Gaussian Model
  '''
  tr('mu', alan.Normal(prior_mean.squeeze(1), prior_scale.squeeze(1)))
  tr('obs', alan.Normal(tr['mu'], lik_scale))



class Q_ml1(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.mu = alan.MLNormal(sample_shape=(num_latents,))

    def forward(self, tr):
        tr('mu', self.mu())

class Q_ml2(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.mu = alan.ML2Normal(sample_shape=(num_latents,))

    def forward(self, tr):
        tr('mu', self.mu())

#Posterior
data = alan.Model(P).sample_prior(varnames='obs')

# prior_scale = t.square(prior_scale)
# lik_scale = t.square(lik_scale)
# post_scale = t.diag(t.diag(prior_scale) @ t.inverse(t.diag(prior_scale) + t.diag(lik_scale)) @ t.diag(lik_scale))
# post_mean = t.diag(prior_scale) @ t.inverse(t.diag(prior_scale) + t.diag(lik_scale)) @ data['obs'] + t.diag(lik_scale) @ t.inverse(t.diag(prior_scale) + t.diag(lik_scale)) @ prior_mean

# post_mean = post_mean.reshape(-1,1)
# post_scale = t.sqrt(post_scale.reshape(-1,1))


# post_params = t.cat([post_mean, post_scale], dim=1)

prior_params = t.cat([prior_mean, prior_scale], dim=1)
lik_params = lik_scale
init = t.tensor([0.0,1.0], dtype=t.float64).repeat((num_latents,1))

# lr = lambda i: ((i + 10)**(-0.9))
seed_torch(0)
m_q, l_one_iters, entropies, times = natural_rws(T, init, 0.01, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None))

seed_torch(0)
m_q_ml1, l_one_iters_ml1, entropies, times = ml1(T, init, 0.01, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None))

seed_torch(0)
m_q_ml2, l_one_iters_ml2, entropies, times = ml2(T, init, 0.01, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None))

seed_torch(0)
q = Q_ml1()
m1 = alan.Model(P, q).condition(data=data)

elbos_ml1 = []
for i in range(T):
    lr = 0.01
    
    sample = m1.sample_same(K, reparam=False)
    elbos_ml1.append(sample.elbo().item()) 


    m1.update(lr, sample)


seed_torch(0)
q = Q_ml2()
m2 = alan.Model(P, q).condition(data=data)

elbos_ml2 = []
for i in range(T):
    lr = 0.01
    
    sample = m2.sample_same(K, reparam=False)
    elbos_ml2.append(sample.elbo().item()) 


    m2.update(lr, sample)


ax_iters.plot(l_one_iters, color=colours[0], label=f'Natural RWS')
ax_iters.plot(l_one_iters_ml1, color=colours[1], label=f'ML1 Toy')
ax_iters.plot(l_one_iters_ml2, color=colours[2], label=f'ML2 Toy')
ax_iters.plot(elbos_ml1, color=colours[3], label=f'ML1', linestyle=':')
ax_iters.plot(elbos_ml2, color=colours[4], label=f'ML2', linestyle=':')
ax_iters.set_xlabel('Iteration')
ax_iters.set_ylabel('ELBO')
ax_iters.legend()
fig_iters.suptitle(f'K={K}, Number of latents={num_latents}')
fig_iters.tight_layout()
fig_iters.savefig(f'figures/ml_test.png')

for i in range(T):
    if np.abs(l_one_iters_ml1[i] - elbos_ml1[i]) > 0.001:
        print('ML1 differs from ML1 toy')
        print(i)
        print(f'ml1: {elbos_ml1[i]}')
        print(f'ml1 toy: {l_one_iters_ml1[i]}')
    if np.abs(l_one_iters_ml2[i] - elbos_ml2[i]) > 0.001:
        print('ML2 differs from ML2 toy')
        print(i)
        print(f'ml2: {elbos_ml2[i]}')
        print(f'ml2 toy: {l_one_iters_ml2[i]}')

