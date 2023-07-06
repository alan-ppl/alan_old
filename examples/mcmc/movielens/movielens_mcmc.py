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

M=1
N=1
sizes = {'plate_1':M, 'plate_2':N}
d_z = 18
covariates = {'x':t.load('examples/mcmc/movielens/data/weights_{0}_{1}.pt'.format(N,M)).to(device)}
data = {'obs':t.load('examples/mcmc/movielens/data/data_y_{0}_{1}.pt'.format(N, M)).to(device)}

# x = t.load('examples/mcmc/movielens/data/weights_{0}_{1}.pt'.format(N,M)).to(device)
# x = t.rand(d_z).to(device)
def P(tr):
  '''
  Bayesian Gaussian Model
  '''
  d = 18
  tr('mu_z', alan.Normal(t.zeros((d_z,)).to(device), t.ones((d_z,)).to(device)))
  tr('psi_z', alan.Normal(t.zeros((d_z,)).to(device), t.ones((d_z,)).to(device)))

  tr('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()))#, plates='plate_1')

  tr('obs', alan.Bernoulli(logits = tr['z'] @ tr['x']))

def Q(tr : TraceQMCMC):  # TraceQMCMC so that we have access to tr.indexed_samples
  # print(tr.platedims)
  # breakpoint()
  for var in tr.indexed_samples:
    tr(var, alan.Normal(tr.indexed_samples[var].rename(None), 0.05))
    # tr(var, alan.Normal(tr.indexed_samples[var], 0.001))


model = alan.Model(P, Q, data, covariates).to(device)
model.check_device(device)

# Get some initial samples
prior_samples = model.sample_prior(1, device=device)
indexed_samples = {key: prior_samples[key].squeeze() for key in prior_samples}
generated_samples = [indexed_samples]

K=2
num_iters = 500
for i in range(num_iters):
    # Obtain new unindexed samples (given current indexed samples)
    sample = model.sample_MCMC(K, indexed_samples, data=None, inputs=None, reparam=False, device=device)

    # Obtain next iteration's indexed samples
    indexed_samples = sample.importance_samples(1)
    for key in indexed_samples:
      val = indexed_samples[key]
      indexed_samples[key] = val.mean(val.dims[0])

    generated_samples.append(indexed_samples)

# print(generated_samples)
# breakpoint()

obs = t.stack([sample["obs"] for sample in generated_samples]).to('cpu')
# print(obs[:,0][100::10], obs[:,1][100::10])
# breakpoint()
# plot
fig, ax = plt.subplots()

ax.scatter(obs[:,0][100::10], obs[:,1][100::10], c=range(num_iters+1)[100::10], cmap='viridis', alpha=0.5)

# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))

plt.savefig("/home/dg22309/Desktop/fig.png")

# breakpoint()
# TODO: make TraceQMCMC inherit from AbstractTraceQ not TraceQPermutation (or whatever)
#        -> will probably require us to deal with logqs in TraceQMCMC?
