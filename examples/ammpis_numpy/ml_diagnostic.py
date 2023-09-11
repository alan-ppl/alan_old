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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-N', type=int, default=5)
parser.add_argument('-K', type=int, default=3)
parser.add_argument('-T', type=int, default=100)
parser.add_argument('-p', '--plot_ML2_only', default=False, action='store_true')
parser.add_argument('-b', '--backup_plots',  default=False, action='store_true')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-m', '--mismatch_count', default=False, action='store_true')
parser.add_argument('-S', '--use_separated_model', default=False, action='store_true')
args = parser.parse_args()

N = args.N
K = args.K
T = args.T
plot_ML2_only = args.plot_ML2_only
backup_plots = args.backup_plots
seed = args.seed
mismatch_count = args.mismatch_count
use_separated_model = args.use_separated_model

# there's some ambiguity in how we want to think about N... (is mu one rv or N rvs?)
# (N rvs seems to be the right way to think about it)
dim_latent = N  

if mismatch_count:
    with open('m_mismatch_count.txt', 'w') as f:
        pass  # clear the file

colours = ['#9e9ac8','#fbb4b9','#253494','#de2d26','#31a354']
fig_iters, ax_iters = plt.subplots(5,1, figsize=(8.0, 25.0))

seed_torch(seed)

prior_mean = Normal(0,150).sample((dim_latent,1)).float()
prior_scale = Uniform(1, 2).sample((dim_latent,1)).float()
lik_scale = Uniform(1, 2).sample((dim_latent,)).float()

def P_old(tr):
    tr('mu', alan.Normal(prior_mean.squeeze(1), prior_scale.squeeze(1)))
    tr('obs', alan.Normal(tr['mu'], lik_scale))

# data = alan.Model(P_old).sample_prior(varnames='obs')
# data['obs'] = data['obs'].rename('plate1')

if use_separated_model:
    covariates = {}
else:
    covariates = {'prior_mean':  prior_mean.clone().squeeze(1),#.rename('plate1'),
                  'prior_scale': prior_scale.clone().squeeze(1),#.rename('plate1'),
                  'lik_scale':   lik_scale.clone()#.rename('plate1')
                 }

def P(tr, prior_mean, prior_scale, lik_scale):
    # the arguments are passed via the inputs=covariates dictionary
    tr('mu', alan.Normal(prior_mean, prior_scale))#, plates="plate1")
    tr('obs', alan.Normal(tr['mu'], lik_scale))#, plates="plate1")
    # print(tr['mu'])

class Q_ml1(alan.AlanModule):
    def __init__(self):
        super().__init__()
        # self.mu = alan.MLNormal(sample_shape=(dim_latent,))
        self.mu = alan.MLNormal(platesizes={"plate1": N}, sample_shape=(1,))

    def forward(self, tr, prior_mean, prior_scale, lik_scale):
        tr('mu', self.mu())

class Q_ml2(alan.AlanModule):
    def __init__(self):
        super().__init__()
        # self.mu = alan.ML2Normal(sample_shape=(dim_latent,))
        self.mu = alan.ML2Normal(platesizes={"plate1": N}, sample_shape=(1,))

    def forward(self, tr, prior_mean, prior_scale, lik_scale):
        tr('mu', self.mu())

def P_separate(tr):
    for i in range(N):
        tr(f'mu{i}', alan.Normal(prior_mean[i], prior_scale[i]))

    mu = t.stack([tr[f'mu{i}'] for i in range(N)]).squeeze(1)

    tr('obs', alan.Normal(mu, lik_scale))

    # print(tr['obs'])
    # print(mu)

class Q_ml_separate(alan.AlanModule):
    def __init__(self):
        super().__init__()
        
        for i in range(N):
            setattr(self, f'mu{i}', alan.MLNormal(sample_shape=(1,)))

    def forward(self, tr):
        for i in range(N):
            tr(f'mu{i}', getattr(self, f'mu{i}')())

class Q_ml2_separate(alan.AlanModule):
    def __init__(self):
        super().__init__()

        for i in range(N):
            setattr(self, f'mu{i}', alan.ML2Normal(sample_shape=(1,)))

    def forward(self, tr):
        for i in range(N):
            tr(f'mu{i}', getattr(self, f'mu{i}')())

if use_separated_model:
    data = alan.Model(P_separate).sample_prior(varnames='obs', inputs={})
else:
    data = alan.Model(P).sample_prior(varnames='obs', platesizes={"plate1": N}, inputs=covariates)


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

# lr = lambda i: 0.01
lr = lambda i: ((i + 10)**(-0.9))

seed_torch(seed)
m_q, l_one_iters, entropies, times = natural_rws(T, init, lr, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None))
print("Natural RWS done.\n")

seed_torch(seed)
m_q_ml1, l_one_iters_ml1, entropies, times = ml1(T, init, lr, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None))
print("ML1 Toy done.\n")

seed_torch(seed)
m_q_ml2, l_one_iters_ml2, entropies, times = ml2(T, init, lr, K, prior_params=prior_params, lik_params=lik_params, data=data['obs'].rename(None))
print("ML2 Toy done.\n")

seed_torch(seed)
if use_separated_model:
    q = Q_ml_separate()
    m1 = alan.Model(P_separate, q).condition(data=data)
else:
    q = Q_ml1()
    m1 = alan.Model(P, q).condition(data=data)

elbos_ml1 = []
for i in range(T):
    sample = m1.sample_same(K, inputs=covariates, reparam=False)
    elbos_ml1.append(sample.elbo().item()) 

    m1.update(lr(i), sample)

print("ML1 done.\n")

seed_torch(seed)
if use_separated_model:
    q = Q_ml2_separate()
    m2 = alan.Model(P_separate, q).condition(data=data)
else:
    q = Q_ml2()
    m2 = alan.Model(P, q).condition(data=data)

elbos_ml2 = []
for i in range(T):
    sample = m2.sample_same(K, inputs=covariates, reparam=False)
    elbos_ml2.append(sample.elbo().item()) 

    m2.update(lr(i), sample)

print("ML2 done.\n")

# Convert elbos to numpy arrays
elbos = [np.array([y.item() if type(y) == t.tensor else y for y in x]) for x in [l_one_iters, l_one_iters_ml1, l_one_iters_ml2, elbos_ml1, elbos_ml2]]

conf_mat = np.ndarray((5,5), dtype=int)
for i in range(5):
    for j in range(5):
        conf_mat[i,j] = sum(np.abs(elbos[i] - elbos[j]) < 0.0001)
# print(conf_mat)

cmap = plt.get_cmap('Greens')
colors = cmap(np.linspace(0.5,1, cmap.N // 2))
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('Upper Half', colors)
im = ax_iters[0].imshow(conf_mat, cmap=cmap2)
# im = ax_iters[4].imshow(conf_mat, cmap='Greens')  # this uses the whole colormap (not just the upper half)

# Add labels to the confusion matrix
ax_iters[0].set_xticks(np.arange(len(conf_mat)))
ax_iters[0].set_yticks(np.arange(len(conf_mat)))
ax_iters[0].set_xticklabels(['Natural RWS', 'ML1 Toy', 'ML2 Toy', 'ML1', 'ML2'])
ax_iters[0].set_yticklabels(['Natural RWS', 'ML1 Toy', 'ML2 Toy', 'ML1', 'ML2'])
plt.setp(ax_iters[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text to the confusion matrix
for i in range(len(conf_mat)):
    for j in range(len(conf_mat)):
        text = ax_iters[0].text(j, i, conf_mat[i, j], ha="center", va="center", color="w")

# Add title
ax_iters[0].set_title("# Iterations where ELBOs are within 0.0001 of each other")

# Add colorbar to the confusion matrix
cbar = ax_iters[0].figure.colorbar(im, ax=ax_iters[0])

print(f'Number of iterations where ML1 and ML1 Toy differ: {T - conf_mat[1,3]}')
print(f'Number of iterations where ML2 and ML2 Toy differ: {T - conf_mat[2,4]}')
print(f'Number of iterations where ML1 and Natural RWS differ: {T - conf_mat[0,3]}')
print(f'Number of iterations where ML2 and Natural RWS differ: {T - conf_mat[0,4]}')
print(f'Number of iterations where ML1 Toy and Natural RWS differ: {T - conf_mat[0,1]}')
print(f'Number of iterations where ML2 Toy and Natural RWS differ: {T - conf_mat[0,2]}')

ax_iters[1].plot(l_one_iters, color=colours[0], label=f'Natural RWS')
ax_iters[1].plot(l_one_iters_ml1, color=colours[1], label=f'ML1 Toy')
ax_iters[1].plot(l_one_iters_ml2, color=colours[2], label=f'ML2 Toy')
ax_iters[1].plot(elbos_ml1, color=colours[3], label=f'ML1', linestyle=':')
ax_iters[1].plot(elbos_ml2, color=colours[4], label=f'ML2', linestyle=':')
ax_iters[1].set_ylabel('ELBO')
ax_iters[1].legend()

if not plot_ML2_only:
    ax_iters[2].plot([(l_one_iters[i] - l_one_iters_ml1[i]) for i in range(T)], color=colours[1], label=f'Natural RWS - ML1 Toy')
    ax_iters[2].plot([(l_one_iters[i] - elbos_ml1[i]) for i in range(T)], color=colours[3], label=f'Natural RWS - ML1', linestyle=':')
ax_iters[2].plot([(l_one_iters[i] - l_one_iters_ml2[i]) for i in range(T)], color=colours[2], label=f'Natural RWS - ML2 Toy')
ax_iters[2].plot([(l_one_iters[i] - elbos_ml2[i]) for i in range(T)], color=colours[4], label=f'Natural RWS - ML2', linestyle=':')
ax_iters[2].set_ylabel('Difference in ELBO')
ax_iters[2].legend()

# if not plot_ML2_only:
#     ax_iters[3].plot([l_one_iters[i] / l_one_iters_ml1[i] for i in range(T)], color=colours[1], label=f'Natural RWS / ML1 Toy')
#     ax_iters[3].plot([l_one_iters[i] / elbos_ml1[i] for i in range(T)], color=colours[3], label=f'Natural RWS / ML1', linestyle=':')
# ax_iters[3].plot([l_one_iters[i] / l_one_iters_ml2[i] for i in range(T)], color=colours[2], label=f'Natural RWS / ML2 Toy')
# ax_iters[3].plot([l_one_iters[i] / elbos_ml2[i] for i in range(T)], color=colours[4], label=f'Natural RWS / ML2', linestyle=':')
ax_iters[3].plot([l_one_iters_ml1[i] / elbos_ml1[i] for i in range(T)], color=colours[1], label=f'ML1 Toy / ML1')
ax_iters[3].plot([l_one_iters_ml2[i] / elbos_ml2[i] for i in range(T)], color=colours[2], label=f'ML2 Toy / ML2')
ax_iters[3].set_ylabel('Ratio of ELBOs')
ax_iters[3].legend()

if not plot_ML2_only:
    ax_iters[4].plot([((m_q[i] - m_q_ml1[i])**2).sum() for i in range(T)], color=colours[1], label=f'||Natural RWS - ML1 Toy||^2')
ax_iters[4].plot([((m_q[i] - m_q_ml2[i])**2).sum() for i in range(T)], color=colours[2], label=f'||Natural RWS - ML2 Toy||^2')
ax_iters[4].set_ylabel('MSE of moments against RWS moments')
ax_iters[4].legend()


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


ax_iters[-1].set_xlabel('Iteration')

fig_iters.suptitle(f'K={K}, Number of latents={dim_latent}', y=0.995, fontsize=16)
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