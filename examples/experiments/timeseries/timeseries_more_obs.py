import torch as t
import torch.distributions as dist
import torch.nn as nn
import alan
import math
from alan.experiment_utils import seed_torch
from functorch.dim import dims, Dim
from alan.utils import *
from alan.dist import Categorical
import json
import numpy as np
import time
seed_torch(2)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

N = 21

var = float(1/math.sqrt(N))
def P(tr):
    tr.sample('ts_1', alan.Normal(0.0, var))
    # print(tr['ts_1'])
    for i in range(2,N+1):
        tr.sample('ts_{}'.format(i), alan.Normal(tr['ts_{}'.format(i-1)], var))
        if i % 3 == 0:
            tr.sample('obs_{}'.format(i//3), alan.Normal(tr['ts_{}'.format(i)], 1))


def Q(tr):
    tr.sample('ts_1', alan.Normal(0.0, var))
    # print(tr['ts_1'])
    for i in range(2,N+1):
        tr.sample('ts_{}'.format(i), alan.Normal(tr['ts_{}'.format(i-1)], var))


# data = {'obs':t.tensor(1.5009)}
data = alan.sample(P, varnames=('obs_1','obs_2','obs_3','obs_4','obs_5','obs_6','obs_7'))
print(data)
model = alan.Model(P, Q, data)
model.to(device)



### PARTICLE FILTER
def pfilter(K):
    #particle filter

    samples = []
    # obs_samples = []
    logps = []

    samples.append(dist.Normal(0, 1/math.sqrt(N)).sample((K,)))
    for i in range(2,N+1):
        if i % 3 == 0:
            samp = dist.Normal(samples[-1], 1/math.sqrt(N)).sample()
            logp = dist.Normal(samp, 1).log_prob(data['obs_{}'.format(i // 3)])
            idxs = dist.Categorical(logits=logp).sample(sample_shape=(K,))
            samp = dist.Normal(samples[-1][idxs], 1/math.sqrt(N)).sample()
            samples.append(samp)
            logps.append(dist.Normal(samples[-1], 1).log_prob(data['obs_{}'.format(i // 3)]))
        else:
            idxs = dist.Categorical(t.ones(K)/K).sample(sample_shape=(K,))
            samples.append(dist.Normal(samples[-1][idxs], 1/math.sqrt(N)).sample())
        # idxs = dist.Categorical(t.ones(K)/K).sample(sample_shape=(K,))


    return t.logsumexp(sum(logps), dim=0) - math.log(K)

## Running exps
Ks=[3,10,30,100,300]
#Ks=[1]
elbos = {k:[] for k in Ks}
elbos_tmc = {k:[] for k in Ks}
elbos_particle = {k:[] for k in Ks}

times = {k:[] for k in Ks}
times_tmc = {k:[] for k in Ks}
times_particle = {k:[] for k in Ks}

for k in Ks:


    # elbos = []
    num_runs = 250
    for i in range(num_runs):
        start = time.time()
        model = alan.Model(P, Q, data)
        elbos[k].append(model.elbo_tmc_new(k).item())
        end = time.time()
        times[k].append(end-start)
        start = time.time()
        model = alan.Model(P, Q, data)
        elbos_tmc[k].append(model.elbo_tmc(k).item())
        end = time.time()
        times_tmc[k].append(end-start)
        start = time.time()
        elbos_particle[k].append(pfilter(k).item())
        end = time.time()
        times_particle[k].append(end-start)

    elbos[k] = {'mean':np.mean(elbos[k]), 'std_err':np.std(elbos[k])/np.sqrt(num_runs), 'time_mean':np.mean(times[k]), 'time_std_err':np.std(times[k])/np.sqrt(num_runs)}
    elbos_tmc[k] = {'mean':np.mean(elbos_tmc[k]), 'std_err':np.std(elbos_tmc[k])/np.sqrt(num_runs), 'time_mean':np.mean(times_tmc[k]), 'time_std_err':np.std(times_tmc[k])/np.sqrt(num_runs)}
    elbos_particle[k] = {'mean':np.mean(elbos_particle[k]), 'std_err':np.std(elbos_particle[k])/np.sqrt(num_runs), 'time_mean':np.mean(times_particle[k]), 'time_std_err':np.std(times_particle[k])/np.sqrt(num_runs)}


file = 'results/timeseries_more_obs_elbo_tmc_new.json'
with open(file, 'w') as f:
    json.dump(elbos, f)

file = 'results/timeseries_more_obs_elbo_tmc.json'
with open(file, 'w') as f:
    json.dump(elbos_tmc, f)

file = 'results/timeseries_more_obs_elbo_particle.json'
with open(file, 'w') as f:
    json.dump(elbos_particle, f)
