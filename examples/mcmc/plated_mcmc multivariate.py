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

K=2
M=2
N=3
platesizes = {'plate_1':M, 'plate_2':N}
d_z = 4
# x = t.load('examples/experiments/movielens/data/weights_{0}_{1}.pt'.format(N,M)).to(device)
x = t.rand((d_z)).to(device)
def P(tr):
  '''
  Bayesian Gaussian Model
  '''
  d = d_z
  tr('mu_x', alan.MultivariateNormal(tr.zeros(d), t.diag(tr.ones(d)))) #, plate="plate_1")
  tr('log_sigma_x', alan.MultivariateNormal(tr.zeros(d), t.diag(tr.ones(d)))) #, plate="plate_1")
  tr('x', alan.MultivariateNormal(tr['mu_x'], t.diag(tr['log_sigma_x'].exp())))#, plates='plate_1')

  tr('mu_y', alan.MultivariateNormal(5*tr.ones(d), t.diag(tr.ones(d))))#, plates="plate_1")
  tr('log_sigma_y', alan.MultivariateNormal(tr.zeros(d), t.diag(tr.ones(d))))#, plates="plate_1")
  print("helooooo", tr['mu_y'], tr['log_sigma_y'])
  tr('y', alan.MultivariateNormal(tr['mu_y'], t.diag(tr['log_sigma_y'].exp())))#, plates='plate_1')

  # if np.random.random() < 0.5:
  #   tr('obs', alan.MultivariateNormal(tr['x'], t.diag(0.0001*tr.ones(d))), plates='plate_1')
  # else:
  #   tr('obs', alan.MultivariateNormal(tr['y'], t.diag(0.0001*tr.ones(d))), plates='plate_1')
  tr('obs', alan.MultivariateNormal(tr['x'] + tr['y'], t.diag(0.25*tr.ones(d))), plates=('plate_1','plate_2'))#, plates='plate_1')

def Q(tr : TraceQMCMC):  # TraceQMCMC so that we have access to tr.indexed_samples
  # breakpoint()
  for var in tr.indexed_samples:
    if var != "obs":
      # tr(var, alan.Normal(tr.indexed_samples[var].rename(None), 0.5))
      print("yay!", tr.indexed_samples[var])
      tr(var, alan.MultivariateNormal(tr.indexed_samples[var], t.diag(0.5*tr.ones(d_z))))
    else:
      # We don't actually specify the plates in Q because tr.indexed_samples[var] already has the plates
      # (so specifying the plates again will lead to duplicate plate dimensions--BAD!)
      #     -> NOT SO SURE ABOUT THIS ^^^ ANYMORE: 
      #       - we need to specify the plates in Q so that dist.log_prob() works when checking self.ndims and stuff
      print("boo!", tr.indexed_samples[var])
      # tr(var, alan.Normal(tr.indexed_samples[var].rename(None), 0.001), plates=('plate_1','plate_2'))
      tr(var, alan.MultivariateNormal(tr.indexed_samples[var], t.diag(0.001*tr.ones(d_z))), plates=('plate_1','plate_2'))
      # tr(var, alan.Categorical((tr.indexed_samples[var])), plates=('plate_1','plate_2'))

model = alan.Model(P, Q).to(device)
model.check_device(device)

# Get some initial samples
prior_samples = model.sample_prior(1, device=device, platesizes=platesizes)
indexed_samples = {key: prior_samples[key].squeeze() for key in prior_samples}
generated_samples = [indexed_samples]
print(generated_samples)
input("MCMC now?")


num_iters = 500
for i in range(num_iters):
    # Obtain new unindexed samples (given current indexed samples)
    sample = model.sample_MCMC(K, indexed_samples, data=None, inputs=None, reparam=False, platesizes=platesizes, device=device)

    # Obtain next iteration's indexed samples
    indexed_samples = sample.importance_samples(1)
    print(indexed_samples)
    input("Press Enter to continue...") 
    for key in indexed_samples:
      val = indexed_samples[key]
      indexed_samples[key] = val.mean(val.dims[0]) # use mean instead of squeeze

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
