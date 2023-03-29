import torch as t
import torch.nn as nn
import alan
import alan.postproc as pp
from alan.traces import *
t.manual_seed(0)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

def P(tr):
  '''
  Bayesian Gaussian Model
  '''
  d = 2
  tr('mu', alan.Normal(t.zeros(d), t.ones(d))) #, plate="plate_1")
  tr('log_sigma', alan.Normal(t.zeros(d), t.ones(d))) #, plate="plate_1")
  tr('obs', alan.MultivariateNormal(tr['mu'], t.diag(tr['log_sigma'].exp())))#, plates='plate_1')

def Q(tr : TraceQMCMC):  # TraceQMCMC so that we have access to tr.indexed_samples
  for var in tr.indexed_samples:
    tr(var, alan.Normal(tr.indexed_samples[var].rename(None), 0.01))
    # tr(var, alan.Normal(tr.indexed_samples[var], 0.001))


model = alan.Model(P, Q)
model.to(device)

# Get some initial samples
prior_samples = model.sample_prior(1)
indexed_samples = {key: prior_samples[key].squeeze() for key in prior_samples}

generated_samples = [indexed_samples]

K=10
for i in range(100):
    # Obtain new unindexed samples (given current indexed samples)
    sample = model.sample_MCMC(K, indexed_samples, data=None, inputs=None, reparam=False, device=device)

    # Obtain next iteration's indexed samples
    indexed_samples = sample.importance_samples(1)
    for key in indexed_samples:
      val = indexed_samples[key]
      indexed_samples[key] = val.mean(val.dims[0])

    generated_samples.append(indexed_samples)

print(generated_samples)

# breakpoint()
# TODO: make TraceQMCMC inherit from AbstractTraceQ not TraceQPermutation (or whatever)
#        -> will probably require us to deal with logqs in TraceQMCMC?
