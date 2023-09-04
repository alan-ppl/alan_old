import torch as t
import torch.nn as nn
import alan
from alan.postproc import *
t.manual_seed(0)

from alan.experiment_utils import seed_torch, n_mean

from torch.distributions import Normal, Uniform
from ammpis import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-N', type=int, default=3)
parser.add_argument('-K', type=int, default=3)
parser.add_argument('-T', type=int, default=100)
parser.add_argument('-p', '--plot_ML2_only', default=False, action='store_true')
args = parser.parse_args()

N = args.N
K = args.K
T = args.T
plot_ML2_only = args.plot_ML2_only

with open('m_mismatch_count.txt', 'w') as f:
    pass

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
colours = ['#9e9ac8','#fbb4b9','#253494','#de2d26','#31a354']

fig_iters, ax_iters = plt.subplots(5,1, figsize=(8.0, 25.0))

seed_torch(0)

dim_latent = N
# K=3

# T=100
prior_mean = Normal(0,150).sample((dim_latent,1)).float()
prior_scale = Uniform(1, 2).sample((dim_latent,1)).float()

lik_scale = Uniform(1, 2).sample((dim_latent,)).float()
def P(tr):
  '''
  Bayesian Gaussian Model
  '''
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
init = t.tensor([0.0,1.0], dtype=t.float64).repeat((dim_latent,1))

# lr = lambda i: ((i + 10)**(-0.9))
seed_torch(0)
m_q, l_one_iters, entropies, times = natural_rws(T, init, 0.01, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None))
print("Natural RWS done.\n")

seed_torch(0)
m_q_ml1, l_one_iters_ml1, entropies, times = ml1(T, init, 0.01, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None))
print("ML1 Toy done.\n")

seed_torch(0)
m_q_ml2, l_one_iters_ml2, entropies, times = ml2(T, init, 0.01, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None))
print("ML2 Toy done.\n")

seed_torch(0)
q = Q_ml1()
m1 = alan.Model(P, q).condition(data=data)

elbos_ml1 = []
for i in range(T):
    lr = 0.01
    
    sample = m1.sample_same(K, reparam=False)
    elbos_ml1.append(sample.elbo().item()) 


    m1.update(lr, sample)

print("ML1 done.\n")

seed_torch(0)
q = Q_ml2()
m2 = alan.Model(P, q).condition(data=data)

elbos_ml2 = []
for i in range(T):
    lr = 0.01
    
    sample = m2.sample_same(K, reparam=False)
    elbos_ml2.append(sample.elbo().item()) 


    m2.update(lr, sample)

print("ML2 done.\n")

ax_iters[0].plot(l_one_iters, color=colours[0], label=f'Natural RWS')
ax_iters[0].plot(l_one_iters_ml1, color=colours[1], label=f'ML1 Toy')
ax_iters[0].plot(l_one_iters_ml2, color=colours[2], label=f'ML2 Toy')
ax_iters[0].plot(elbos_ml1, color=colours[3], label=f'ML1', linestyle=':')
ax_iters[0].plot(elbos_ml2, color=colours[4], label=f'ML2', linestyle=':')
ax_iters[0].set_ylabel('ELBO')
ax_iters[0].legend()

if not plot_ML2_only:
    ax_iters[1].plot([(l_one_iters[i] - l_one_iters_ml1[i]) for i in range(T)], color=colours[1], label=f'Natural RWS - ML1 Toy')
    ax_iters[1].plot([(l_one_iters[i] - elbos_ml1[i]) for i in range(T)], color=colours[3], label=f'Natural RWS - ML1', linestyle=':')
ax_iters[1].plot([(l_one_iters[i] - l_one_iters_ml2[i]) for i in range(T)], color=colours[2], label=f'Natural RWS - ML2 Toy')
ax_iters[1].plot([(l_one_iters[i] - elbos_ml2[i]) for i in range(T)], color=colours[4], label=f'Natural RWS - ML2', linestyle=':')
ax_iters[1].set_ylabel('Difference in ELBO')
ax_iters[1].legend()

if not plot_ML2_only:
    ax_iters[2].plot([l_one_iters[i] / l_one_iters_ml1[i] for i in range(T)], color=colours[1], label=f'Natural RWS / ML1 Toy')
    ax_iters[2].plot([l_one_iters[i] / elbos_ml1[i] for i in range(T)], color=colours[3], label=f'Natural RWS / ML1', linestyle=':')
ax_iters[2].plot([l_one_iters[i] / l_one_iters_ml2[i] for i in range(T)], color=colours[2], label=f'Natural RWS / ML2 Toy')
ax_iters[2].plot([l_one_iters[i] / elbos_ml2[i] for i in range(T)], color=colours[4], label=f'Natural RWS / ML2', linestyle=':')
ax_iters[2].set_ylabel('Ratio of ELBOs')
ax_iters[2].legend()

if not plot_ML2_only:
    ax_iters[3].plot([((m_q[i] - m_q_ml1[i])**2).sum() for i in range(T)], color=colours[1], label=f'||Natural RWS - ML1 Toy||^2')
ax_iters[3].plot([((m_q[i] - m_q_ml2[i])**2).sum() for i in range(T)], color=colours[2], label=f'||Natural RWS - ML2 Toy||^2')
ax_iters[3].set_ylabel('MSE of moments')
ax_iters[3].legend()


l_one_iters_diffs = np.array([np.nan] + [l_one_iters[i+1] - l_one_iters[i] for i in range(T-1)])
l_one_iters_ml1_diffs = np.array([np.nan] + [l_one_iters_ml1[i+1] - l_one_iters_ml1[i] for i in range(T-1)])
l_one_iters_ml2_diffs = np.array([np.nan] + [l_one_iters_ml2[i+1] - l_one_iters_ml2[i] for i in range(T-1)])
elbos_ml1_diffs = np.array([np.nan] + [elbos_ml1[i+1] - elbos_ml1[i] for i in range(T-1)])
elbos_ml2_diffs = np.array([np.nan] + [elbos_ml2[i+1] - elbos_ml2[i] for i in range(T-1)])

# ax_iters[4].plot(, color=colours[0], label=f'Natural RWS')
if not plot_ML2_only:
    ax_iters[4].plot(l_one_iters_ml1_diffs - l_one_iters_diffs, color=colours[1], label=f'ML1 Toy')
    ax_iters[4].plot(elbos_ml1_diffs - l_one_iters_diffs, color=colours[3], label=f'ML1', linestyle=':')
ax_iters[4].plot(l_one_iters_ml2_diffs - l_one_iters_diffs, color=colours[2], label=f'ML2 Toy')
ax_iters[4].plot(elbos_ml2_diffs - l_one_iters_diffs, color=colours[4], label=f'ML2', linestyle=':')
ax_iters[4].set_ylabel('Difference in ELBO step-sizes against natural_rws')
ax_iters[4].legend()

ax_iters[-1].set_xlabel('Iteration')

fig_iters.suptitle(f'K={K}, Number of latents={dim_latent}')
fig_iters.tight_layout()
fig_iters.savefig(f'figures/ml_diagnostics/N{N}_K{K}_T{T}{"noML1" if plot_ML2_only else ""}.png')
fig_iters.savefig(f'figures/ml_diagnostic.png')
plt.close()

diff_ml1_ml1toy = 0
diff_ml2_ml2toy = 0
diff_ml1_natural = 0
diff_ml2_natural = 0
for i in range(T):
    if np.abs(l_one_iters_ml1[i] - elbos_ml1[i]) > 0.001:
        diff_ml1_ml1toy += 1
    if np.abs(l_one_iters_ml2[i] - elbos_ml2[i]) > 0.001:
        diff_ml2_ml2toy += 1
    if np.abs(l_one_iters[i] - elbos_ml1[i]) > 0.001:
        diff_ml1_natural += 1
    if np.abs(l_one_iters[i] - elbos_ml2[i]) > 0.001:
        diff_ml2_natural += 1

print(f'Number of iterations where ML1 and ML1 Toy differ: {diff_ml1_ml1toy}')
print(f'Number of iterations where ML2 and ML2 Toy differ: {diff_ml2_ml2toy}')
print(f'Number of iterations where ML1 and Natural RWS differ: {diff_ml1_natural}')
print(f'Number of iterations where ML2 and Natural RWS differ: {diff_ml2_natural}')


# open the file m_mismatch_count.txt and plot the numbers in each line of the file
with open('m_mismatch_count.txt', 'r') as f:
    lines = f.readlines()
    numbers = [int(line.strip()) for line in lines]

# Create a figure and an axes object
fig, ax = plt.subplots()

# Plot the numbers on the axes object
ax.plot(numbers)

# Set the x and y labels
ax.set_xlabel('Iteration')
ax.set_ylabel('Count')
ax.set_title(f'Number of mismatched m_new entries\n(between RWS and ML2)\nN={N}, K={K}')

# fig.savefig(f"figures/m_mismatch_countN{N}_K{K}_T{T}.png")
fig.savefig(f"figures/m_mismatch_count.png")
plt.close()

# for i in range(T):
#     print(f"{i}\n{m_q[i]}\n{m_q_ml2[i]}\n{m_q_ml2[i]}\n")
#     input()

# get moment updates (before lr multiplication) from each iteration
updates_rws = [init] + [(m_q[i+1] - m_q[i]*(1-lr))/lr for i in range(T-1)]
updates_ml1 = [init] + [(m_q_ml1[i+1] - m_q_ml1[i]*(1-lr))/lr for i in range(T-1)]
updates_ml2 = [init] + [(m_q_ml2[i+1] - m_q_ml2[i]*(1-lr))/lr for i in range(T-1)]

# next_command = input("Press enter to continue, b for breakpoint, or q to quit: ")
# if next_command != 'q':
#     if next_command == 'b':
#         breakpoint()
#     for i in range(T):
#         print(f"{i}\n{updates_rws[i]}\n{updates_ml1[i]}\n{updates_ml2[i]}\n")
#         next_command = input("Press enter to continue, b for breakpoint, or q to quit: ")
#         if next_command == 'q':
#             break
#         elif next_command == 'b':
#             breakpoint()