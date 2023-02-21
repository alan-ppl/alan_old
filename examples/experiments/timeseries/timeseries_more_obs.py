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
seed_torch(1)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

N = 30

var = float(1/math.sqrt(N))
tau = 5
def P(tr):
    tr.sample('ts_1', alan.Normal(0.0, np.sqrt(2/tau))))
    # print(tr['ts_1'])
    for i in range(2,N+1):
        latent = (1-(1/tau))*tr['ts_{}'.format(i-1)]
        tr.sample('ts_{}'.format(i), alan.Normal(latent, np.sqrt(2/tau)))
        if i % 3 == 0:
            tr.sample('obs_{}'.format(i//3), alan.Normal(tr['ts_{}'.format(i)], 1))
        # if i == 20:
        #     tr.sample('obs_{}'.format((N+1)//7), alan.Normal(tr['ts_{}'.format(i)], 1))

def Q(tr):
    tr.sample('ts_1', alan.Normal(0.0, np.sqrt(2/tau))))
    # print(tr['ts_1'])
    for i in range(2,N+1):
        latent = (1-(1/tau))*tr['ts_{}'.format(i-1)]
        tr.sample('ts_{}'.format(i), alan.Normal(latent, np.sqrt(2/tau)))


# data = {'obs':t.tensor(1.5009)}
# data = alan.sample(P, varnames=('obs_1','obs_2','obs_3','obs_4','obs_5', 'obs_6', 'obs_7'))
# data = {'obs_1':t.tensor(1), 'obs_2':t.tensor(3), 'obs_3':t.tensor(5), 'obs_4':t.tensor(7),
#         'obs_5':t.tensor(9), 'obs_6':t.tensor(11), 'obs_7':t.tensor(13)}




### PARTICLE FILTER
def pfilter(K):
    #particle filter

    samples = []
    # obs_samples = []
    logps = []

    samples.append(dist.Normal(0, np.sqrt(2/tau))).sample((K,)))
    for i in range(2,N+1):
        if i % 3 == 0:
            samp = dist.Normal((1-(1/tau))*samples[-1], np.sqrt(2/tau)).sample()
            logp = dist.Normal(samp, 1).log_prob(data['obs_{}'.format(i // 3)])
            logps.append(t.logsumexp(logp, dim=0) - math.log(K))
            idxs = dist.Categorical(logits=logp).sample(sample_shape=(K,))
            samp = dist.Normal((1-(1/tau))*samples[-1][idxs], np.sqrt(2/tau)).sample()
            samples.append(samp)
        # elif i==N:
        #     samp = dist.Normal(samples[-1], var).sample()
        #     logp = dist.Normal(samp, 1).log_prob(data['obs_{}'.format((N+1)//7)])
        #     logps.append(t.logsumexp(logp, dim=0) - math.log(K))
        #     idxs = dist.Categorical(logits=logp).sample(sample_shape=(K,))
        #     samp = dist.Normal(samples[-1][idxs], var).sample()
        #     samples.append(samp)
        else:
            idxs = dist.Categorical(t.ones(K)/K).sample(sample_shape=(K,))
            samples.append(dist.Normal((1-(1/tau))*samples[-1][idxs], np.sqrt(2/tau)).sample())
        # idxs = dist.Categorical(t.ones(K)/K).sample(sample_shape=(K,))

    num_obs = len(logps)

    return sum(logps) #- math.log(K**num_obs)

## Running exps
Ks=[3,10,30,100,300,1000]
#Ks=[1]
elbos = {k:[] for k in Ks}
elbos_tmc = {k:[] for k in Ks}
elbos_particle = {k:[] for k in Ks}
elbos_global = {k:[] for k in Ks}


times = {k:[] for k in Ks}
times_tmc = {k:[] for k in Ks}
times_particle = {k:[] for k in Ks}
times_global = {k:[] for k in Ks}

varnames = ('obs_{}'.format(j) for j in range(1,N//3 + 1))
num_runs = 100
for i in range(num_runs):
    varnames = ('obs_{}'.format(j) for j in range(1,N//3 + 1))
    data = alan.sample(P, varnames=tuple(varnames))
    for k in Ks:
        model = alan.Model(P, Q, data)
        start = time.time()
        elbos[k].append(model.elbo_tmc_new(k).item())
        end = time.time()
        times[k].append(end-start)
        start = time.time()
        elbos_tmc[k].append(model.elbo_tmc(k).item())
        end = time.time()
        times_tmc[k].append(end-start)
        start = time.time()
        elbos_particle[k].append(pfilter(k).item())
        end = time.time()
        times_particle[k].append(end-start)
        start = time.time()
        elbos_global[k].append(model.elbo_global(k).item())
        end = time.time()
        times_global[k].append(end-start)

for k in Ks:
    elbos[k] = {'mean':np.mean(elbos[k]), 'std_err':np.std(elbos[k])/np.sqrt(num_runs), 'time_mean':np.mean(times[k]), 'time_std_err':np.std(times[k])/np.sqrt(num_runs)}
    elbos_tmc[k] = {'mean':np.mean(elbos_tmc[k]), 'std_err':np.std(elbos_tmc[k])/np.sqrt(num_runs), 'time_mean':np.mean(times_tmc[k]), 'time_std_err':np.std(times_tmc[k])/np.sqrt(num_runs)}
    elbos_particle[k] = {'mean':np.mean(elbos_particle[k]), 'std_err':np.std(elbos_particle[k])/np.sqrt(num_runs), 'time_mean':np.mean(times_particle[k]), 'time_std_err':np.std(times_particle[k])/np.sqrt(num_runs)}
    elbos_global[k] = {'mean':np.mean(elbos_global[k]), 'std_err':np.std(elbos_global[k])/np.sqrt(num_runs), 'time_mean':np.mean(times_global[k]), 'time_std_err':np.std(times_global[k])/np.sqrt(num_runs)}

file = 'results/timeseries_more_obs_elbo_tmc_new.json'
with open(file, 'w') as f:
    json.dump(elbos, f)

file = 'results/timeseries_more_obs_elbo_tmc.json'
with open(file, 'w') as f:
    json.dump(elbos_tmc, f)

file = 'results/timeseries_more_obs_elbo_particle.json'
with open(file, 'w') as f:
    json.dump(elbos_particle, f)

file = 'results/timeseries_more_obs_elbo_global.json'
with open(file, 'w') as f:
    json.dump(elbos_global, f)
