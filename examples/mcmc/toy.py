import torch as t
import torch.nn as nn
import alan
import alan.postproc as pp
from alan.traces import *
import matplotlib.pyplot as plt
import numpy as np
t.manual_seed(0)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(device)

K = 2
num_samples = 5000#0
slice_step = 1#25
burn_in = 0

num_runs = num_samples*slice_step + burn_in -1

d = 2

def P(tr):
  '''
  Bayesian Gaussian Model
  '''
  
  tr('x', alan.Normal(tr.zeros(d), tr.ones(d)))
  tr('y', alan.Normal(tr.zeros(d), tr.ones(d)))
  tr('obs', alan.Normal(tr['x'], tr['y'].exp()))
#   tr('obs', alan.MultivariateNormal(tr['x'], 0.25*t.eye(d).to(device)))#, plates='plate_1')

def get_Q(q_var = 0.05):
   
    def Q(tr : TraceQMCMC):  # TraceQMCMC so that we have access to tr.indexed_samples
        # print(tr.platedims)
        # breakpoint()
        for var in tr.indexed_samples:
            # print(var)
            tr(var, alan.Normal(tr.indexed_samples[var].rename(None), q_var))

    return Q


q_vars = [0.01, 0.02, 0.035, 0.05]#, 0.1]

results = {str(q_var): {"x": {}, "y": {}} for q_var in q_vars}

for i, q_var in enumerate(q_vars):
    # t.manual_seed(i)

    model = alan.Model(P, get_Q(q_var)).to(device)
    model.check_device(device)

    # Get true underlying data
    sampledData = model.sample_prior(1, device=device)
    data = {'obs': sampledData['obs']}
    # print(sampledData)

    # Get some initial samples
    prior_samples = model.sample_prior(1, device=device)
    indexed_samples = {key: prior_samples[key].squeeze() for key in prior_samples}
    generated_samples = [indexed_samples]

    # Run MCMC
    for _ in range(num_runs):
        # Obtain new unindexed samples (given current indexed samples)
        sample = model.sample_MCMC(K, indexed_samples, data=data, inputs=None, reparam=False, device=device)

        indexed_samples = sample.importance_samples(1)
        for key in indexed_samples:
            val = indexed_samples[key]
            indexed_samples[key] = val.mean(val.dims[0]) # mean instead of squeeze

        generated_samples.append(indexed_samples)

    xs = t.stack([sample["x"] for sample in generated_samples]).to('cpu')
    ys = t.stack([sample["y"] for sample in generated_samples]).to('cpu')

    for key in sampledData:
        sampledData[key] = sampledData[key].cpu()
        if key != 'obs':
            for s in generated_samples:
                s[key] = s[key].cpu()

    x_samples = xs[burn_in::slice_step,:]
    y_samples = ys[burn_in::slice_step,:]

    true_x = sampledData['x'][0,:].rename(None)
    true_y = sampledData['y'][0,:].rename(None)

    estm_x = x_samples.mean(0)
    estm_y = y_samples.mean(0)

    print(f"Run {i+1} of {len(q_vars)}: q_var = {q_var}")
    print(f"Mean sampled x: {estm_x}")
    print(f"True x: {true_x}")
    print(f"Mean sampled x: {estm_y}")
    print(f"True x: {true_y}")
    print()

    results[str(q_var)]["x"] = {"true": true_x, "estm": estm_x, "mses": ((x_samples - sampledData['x'])**2).sum(1).rename(None), "samples": x_samples}
    results[str(q_var)]["y"] = {"true": true_y, "estm": estm_y, "mses": ((y_samples - sampledData['y'])**2).sum(1).rename(None), "samples": y_samples}

    # breakpoint()

q_vars = [str(q_var) for q_var in q_vars]

# Trace Plots
fig, ax = plt.subplots(2,4,figsize=(8.5, 4.5))
for i, q_var in enumerate(q_vars):
    for j, key in enumerate(results[q_var]):
        true = results[q_var][key]["true"]
        estm = results[q_var][key]["estm"]
        mses = results[q_var][key]["mses"]
        samples = results[q_var][key]["samples"]

        sc = ax[j,i].scatter(samples[:,0], samples[:,1], c=range(len(samples)), cmap='viridis', alpha=0.5)

        ax[j,i].scatter(true[0].item(), true[1].item(), c='red', marker='x', s=100, label=f'True Value')
        ax[j,i].scatter(estm[0].item(), estm[1].item(), c='blue', marker='^', s=100, label=f'Mean Sampled Value')

    ax[0,i].set_title(f"c = {q_var}")
ax[0,0].set_ylabel(f"x")
ax[1,0].set_ylabel(f"y")

plt.legend(bbox_to_anchor=(0.1, -0.05), loc="lower left",
                bbox_transform=fig.transFigure, ncol=2)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax[1,3],
                    width="100%",  
                    height="5%",
                    loc='lower center',
                    borderpad=-4
                   )
fig.colorbar(sc, cax=axins, orientation="horizontal", label="Sample Number")

# fig.colorbar(sc, orientation="horizontal", pad=-2)

plt.savefig(f"traces.pdf",bbox_inches="tight")


# MSE Plots
fig, ax = plt.subplots(2,4,figsize=(8.5, 4.5), sharey="row", sharex="col")
for i, q_var in enumerate(q_vars):
    for j, key in enumerate(results[q_var]):
        mses = results[q_var][key]["mses"]
        ax[j,i].plot(mses, label=f"q_var = {q_var}")

    ax[0,i].set_title(f"c = {q_var}")
ax[0,0].set_ylabel(f"MSE of x")
ax[1,0].set_ylabel(f"MSE of y")

# plt.legend(bbox_to_anchor=(0.5, -0.05), loc="lower center",
#                 bbox_transform=fig.transFigure, ncol=4)
plt.savefig(f"mse.pdf", bbox_inches="tight")


# Autocorrelation Plots
fig, ax = plt.subplots(4,4,figsize=(8.5, 4.5), sharex="col", sharey="row")
for j_, key in enumerate(["x", "y"]):
    for k in [0, 1]:
        j = 2*j_ + k

        for i, q_var in enumerate(q_vars):
            samples = results[q_var][key]["samples"]
            ax[0,i].set_title(f"c = {q_var}")

            samples = results[q_var][key]["samples"]
            normalised = samples - samples.mean(0)
            n = normalised.shape[0]
            # breakpoint()
            autocorr = np.correlate(normalised[:,k], normalised[:,k], mode='full')[n-1:] / (n * normalised[:,k].var(0))    
            ax[j,i].plot(autocorr)
            ax[j,i].plot(np.zeros(num_samples), linestyle=(0, (5, 10)), color="grey", alpha=0.5)

            if j == 3:
                ax[j,i].set_xlabel(f"lag")
            else:
                ax[j,i].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off

        ax[j,0].set_ylabel(f"{key}_{k+1}")

plt.savefig(f"autocorr.pdf", bbox_inches="tight")

breakpoint()