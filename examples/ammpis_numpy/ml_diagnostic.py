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
parser.add_argument('-b', '--backup_plots',  default=False, action='store_true')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-m', '--mismatch_count', default=False, action='store_true')
args = parser.parse_args()

N = args.N
K = args.K
T = args.T
plot_ML2_only = args.plot_ML2_only
backup_plots = args.backup_plots
seed = args.seed
mismatch_count = args.mismatch_count

if mismatch_count:
    with open('m_mismatch_count.txt', 'w') as f:
        pass

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
colours = ['#9e9ac8','#fbb4b9','#253494','#de2d26','#31a354']

fig_iters, ax_iters = plt.subplots(5,1, figsize=(8.0, 25.0))

seed_torch(seed)

dim_latent = N
# K=3

# T=100
prior_mean = Normal(0,150).sample((dim_latent,1)).float()
prior_scale = Uniform(1, 2).sample((dim_latent,1)).float()
lik_scale = Uniform(1, 2).sample((dim_latent,)).float()

# # uncomment to use plate of size N rather than N fully distinct latent variables
# prior_mean = t.zeros((N,1))
# prior_scale = t.ones((N,1))
# lik_scale = t.ones(N)

def P(tr):
    '''
    Bayesian Gaussian Model
    '''
    tr('mu', alan.Normal(prior_mean.squeeze(1), prior_scale.squeeze(1)))
    tr('obs', alan.Normal(tr['mu'], lik_scale))

    # # use below lines instead of the above for plated model
    # tr('mu', alan.Normal(0, 1), plates="plate1")
    # tr('obs', alan.Normal(tr['mu'], 1))

    # print(tr['mu'])

class Q_ml1(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.mu = alan.MLNormal(sample_shape=(dim_latent,))
        # self.mu = alan.MLNormal(platesizes={"plate1": N})

    def forward(self, tr):
        tr('mu', self.mu())

class Q_ml2(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.mu = alan.ML2Normal(sample_shape=(dim_latent,))
        # self.mu = alan.ML2Normal(platesizes={"plate1": N})

    def forward(self, tr):
        tr('mu', self.mu())

#Posterior
data = alan.Model(P).sample_prior(varnames='obs')
# data = alan.Model(P).sample_prior(varnames='obs', platesizes={"plate1": N})

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
seed_torch(seed)
m_q, l_one_iters, entropies, times = natural_rws(T, init, 0.01, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None))
print("Natural RWS done.\n")

seed_torch(seed)
m_q_ml1, l_one_iters_ml1, entropies, times = ml1(T, init, 0.01, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None))
print("ML1 Toy done.\n")

seed_torch(seed)
m_q_ml2, l_one_iters_ml2, entropies, times = ml2(T, init, 0.01, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None))
print("ML2 Toy done.\n")

seed_torch(seed)
q = Q_ml1()
m1 = alan.Model(P, q).condition(data=data)

elbos_ml1 = []
for i in range(T):
    lr = 0.01
    
    sample = m1.sample_same(K, reparam=False)
    elbos_ml1.append(sample.elbo().item()) 


    m1.update(lr, sample)

print("ML1 done.\n")

# nasty ml2 hack (for N=3)
#   by treating each latent variable separately, I think the issues relating to `sum_non_dim`
#   in `extra_log_factor` are dealt with in a way that _might_ mean ML2 behaves like ML2_toy 
#   the problem is:
#       1. by treating each latent variable separately, we mess up the random seeding (I'm pretty sure)
#       2. this is suboptimal specification for an alan model
#               (although hopefully if we end up changing alan to fix the `sum_non_dim` stuff, 
#                we won't have to do this weird model specification anymore)
#
#   although the elbos of ML2 and ML2_toy (= rws) are still different, the performances _seem_ comparable
def P_separate(tr):
    '''
    Bayesian Gaussian Model
    '''
    # breakpoint()
    tr('mu1', alan.Normal(prior_mean[0], prior_scale[0]))
    tr('mu2', alan.Normal(prior_mean[1], prior_scale[1]))
    tr('mu3', alan.Normal(prior_mean[2], prior_scale[2]))

    mu = t.stack([tr['mu1'], tr['mu2'], tr['mu3']])
    # print(mu.squeeze(1))

    tr('obs', alan.Normal(mu.squeeze(1), lik_scale))

class Q_ml2_separate(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.mu1 = alan.ML2Normal(sample_shape=(1,))
        self.mu2 = alan.ML2Normal(sample_shape=(1,))
        self.mu3 = alan.ML2Normal(sample_shape=(1,))

    def forward(self, tr):
        tr('mu1', self.mu1())
        tr('mu2', self.mu2())
        tr('mu3', self.mu3())

seed_torch(seed)
# q = Q_ml2()
# m2 = alan.Model(P, q).condition(data=data)

q = Q_ml2_separate()
m2 = alan.Model(P_separate, q).condition(data=data)

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

# if not plot_ML2_only:
#     ax_iters[2].plot([l_one_iters[i] / l_one_iters_ml1[i] for i in range(T)], color=colours[1], label=f'Natural RWS / ML1 Toy')
#     ax_iters[2].plot([l_one_iters[i] / elbos_ml1[i] for i in range(T)], color=colours[3], label=f'Natural RWS / ML1', linestyle=':')
# ax_iters[2].plot([l_one_iters[i] / l_one_iters_ml2[i] for i in range(T)], color=colours[2], label=f'Natural RWS / ML2 Toy')
# ax_iters[2].plot([l_one_iters[i] / elbos_ml2[i] for i in range(T)], color=colours[4], label=f'Natural RWS / ML2', linestyle=':')
ax_iters[2].plot([l_one_iters_ml1[i] / elbos_ml1[i] for i in range(T)], color=colours[1], label=f'ML1 Toy / ML1')
ax_iters[2].plot([l_one_iters_ml2[i] / elbos_ml2[i] for i in range(T)], color=colours[2], label=f'ML2 Toy / ML2')
ax_iters[2].set_ylabel('Ratio of ELBOs')
ax_iters[2].legend()

if not plot_ML2_only:
    ax_iters[3].plot([((m_q[i] - m_q_ml1[i])**2).sum() for i in range(T)], color=colours[1], label=f'||Natural RWS - ML1 Toy||^2')
ax_iters[3].plot([((m_q[i] - m_q_ml2[i])**2).sum() for i in range(T)], color=colours[2], label=f'||Natural RWS - ML2 Toy||^2')
ax_iters[3].set_ylabel('MSE of moments against RWS moments')
ax_iters[3].legend()


# l_one_iters_diffs = np.array([np.nan] + [l_one_iters[i+1] - l_one_iters[i] for i in range(T-1)])
# l_one_iters_ml1_diffs = np.array([np.nan] + [l_one_iters_ml1[i+1] - l_one_iters_ml1[i] for i in range(T-1)])
# l_one_iters_ml2_diffs = np.array([np.nan] + [l_one_iters_ml2[i+1] - l_one_iters_ml2[i] for i in range(T-1)])
# elbos_ml1_diffs = np.array([np.nan] + [elbos_ml1[i+1] - elbos_ml1[i] for i in range(T-1)])
# elbos_ml2_diffs = np.array([np.nan] + [elbos_ml2[i+1] - elbos_ml2[i] for i in range(T-1)])

# ax_iters[4].plot(, color=colours[0], label=f'Natural RWS')
# if not plot_ML2_only:
#     ax_iters[4].plot(l_one_iters_ml1_diffs - l_one_iters_diffs, color=colours[1], label=f'ML1 Toy')
#     ax_iters[4].plot(elbos_ml1_diffs - l_one_iters_diffs, color=colours[3], label=f'ML1', linestyle=':')
# ax_iters[4].plot(l_one_iters_ml2_diffs - l_one_iters_diffs, color=colours[2], label=f'ML2 Toy')
# ax_iters[4].plot(elbos_ml2_diffs - l_one_iters_diffs, color=colours[4], label=f'ML2', linestyle=':')
# ax_iters[4].set_ylabel('Difference in ELBO step-sizes against natural_rws')
# ax_iters[4].legend()


# ax_iters[4].plot([((prior_params - m_q[i])**2).sum() for i in range(T)], color=colours[0], label=f'Natural RWS')
# if not plot_ML2_only:
#     ax_iters[4].plot([((prior_params - m_q_ml1[i])**2).sum() for i in range(T)], color=colours[1], label=f'ML1 Toy')
# ax_iters[4].plot([((prior_params - m_q_ml2[i])**2).sum() for i in range(T)], color=colours[2], label=f'ML2 Toy')
# ax_iters[4].set_ylabel('MSE of moments against prior moments')
# ax_iters[4].legend()


# Convert elbos to numpy arrays
elbos = [np.array([y.item() if type(y) == t.tensor else y for y in x]) for x in [l_one_iters, l_one_iters_ml1, l_one_iters_ml2, elbos_ml1, elbos_ml2]]

conf_mat = np.ndarray((5,5), dtype=int)
for i in range(5):
    for j in range(5):
        conf_mat[i,j] = sum(np.abs(elbos[i] - elbos[j]) < 0.001)
# print(conf_mat)

im = ax_iters[4].imshow(conf_mat, cmap='Greens')

# Add labels to the confusion matrix
ax_iters[4].set_xticks(np.arange(len(conf_mat)))
ax_iters[4].set_yticks(np.arange(len(conf_mat)))
ax_iters[4].set_xticklabels(['Natural RWS', 'ML1 Toy', 'ML2 Toy', 'ML1', 'ML2'])
ax_iters[4].set_yticklabels(['Natural RWS', 'ML1 Toy', 'ML2 Toy', 'ML1', 'ML2'])
plt.setp(ax_iters[4].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text to the confusion matrix
for i in range(len(conf_mat)):
    for j in range(len(conf_mat)):
        text = ax_iters[4].text(j, i, conf_mat[i, j], ha="center", va="center", color="w")

# Add title
ax_iters[4].set_title("# Iterations where ELBOs are within 0.001 of each other")

# Add colorbar to the confusion matrix
cbar = ax_iters[4].figure.colorbar(im, ax=ax_iters[4])

print(f'Number of iterations where ML1 and ML1 Toy differ: {T - conf_mat[1,3]}')
print(f'Number of iterations where ML2 and ML2 Toy differ: {T - conf_mat[2,4]}')
print(f'Number of iterations where ML1 and Natural RWS differ: {T - conf_mat[0,3]}')
print(f'Number of iterations where ML2 and Natural RWS differ: {T - conf_mat[0,4]}')
print(f'Number of iterations where ML1 Toy and Natural RWS differ: {T - conf_mat[0,1]}')
print(f'Number of iterations where ML2 Toy and Natural RWS differ: {T - conf_mat[0,2]}')


ax_iters[-2].set_xlabel('Iteration')

fig_iters.suptitle(f'K={K}, Number of latents={dim_latent}')
fig_iters.tight_layout()
if backup_plots:
    fig_iters.savefig(f'figures/ml_diagnostics/N{N}_K{K}_T{T}{"noML1" if plot_ML2_only else ""}.png')
fig_iters.savefig(f'figures/ml_diagnostic.png')
plt.close()



if mismatch_count:
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

    if backup_plots:
        fig.savefig(f"figures/ml_diagnostics/m_mismatch_count/N{N}_K{K}_T{T}.png")
    fig.savefig(f"figures/m_mismatch_count.png")
    plt.close()

# for i in range(T):
#     print(f"{i}\n{m_q[i]}\n{m_q_ml2[i]}\n{m_q_ml2[i]}\n")
#     input()

# # get moment updates (before lr multiplication) from each iteration
# updates_rws = [init] + [(m_q[i+1] - m_q[i]*(1-lr))/lr for i in range(T-1)]
# updates_ml1 = [init] + [(m_q_ml1[i+1] - m_q_ml1[i]*(1-lr))/lr for i in range(T-1)]
# updates_ml2 = [init] + [(m_q_ml2[i+1] - m_q_ml2[i]*(1-lr))/lr for i in range(T-1)]

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