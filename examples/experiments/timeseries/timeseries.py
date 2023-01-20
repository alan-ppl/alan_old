import torch as t
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


N = 51

var = float(1/math.sqrt(N))
def P(tr):
    tr.sample('ts_1', alan.Normal(0.0, var))
    # print(tr['ts_1'])
    for i in range(2,N+1):
        tr.sample('ts_{}'.format(i), alan.Normal(tr['ts_{}'.format(i-1)], var))
        # print(tr['ts_{}'.format(i)])
    tr.sample('obs', alan.Normal(tr['ts_{}'.format(N)], 1.0))

def Q(tr):
    tr.sample('ts_1', alan.Normal(0.0, var))
    # print(tr['ts_1'])
    for i in range(2,N+1):
        tr.sample('ts_{}'.format(i), alan.Normal(tr['ts_{}'.format(i-1)], var))


data = alan.sample(P, varnames=('obs',))
model = alan.Model(P, Q, data)
model.to(device)
Ks=[1,3,10,30,300]
#Ks=[1]
elbos = {k:[] for k in Ks}
elbos_tmc = {k:[] for k in Ks}
elbos_global = {k:[] for k in Ks}

times = {k:[] for k in Ks}
times_tmc = {k:[] for k in Ks}
times_global = {k:[] for k in Ks}

print(data)
for k in Ks:


    # elbos = []
    num_runs = 1000
    for i in range(num_runs):
        start = time.time()
        elbos[k].append(model.elbo_tmc_new(k).item()/num_runs)
        end = time.time()
        times[k].append(end-start)
        start = time.time()
        elbos_tmc[k].append(model.elbo_tmc(k).item()/num_runs)
        end = time.time()
        times_tmc[k].append(end-start)
        start = time.time()
        elbos_global[k].append(model.elbo_global(k).item()/num_runs)
        end = time.time()
        times_global[k].append(end-start)

    elbos[k] = {'mean':np.mean(elbos[k]), 'std_err':np.std(elbos[k])/np.sqrt(num_runs), 'time_mean':np.mean(times[k]), 'time_std_err':np.std(times[k])/np.sqrt(num_runs)}
    elbos_tmc[k] = {'mean':np.mean(elbos_tmc[k]), 'std_err':np.std(elbos_tmc[k])/np.sqrt(num_runs), 'time_mean':np.mean(times_tmc[k]), 'time_std_err':np.std(times_tmc[k])/np.sqrt(num_runs)}
    elbos_global[k] = {'mean':np.mean(elbos_global[k]), 'std_err':np.std(elbos_global[k])/np.sqrt(num_runs), 'time_mean':np.mean(times_global[k]), 'time_std_err':np.std(times_global[k])/np.sqrt(num_runs)}

file = 'results/timeseries_elbo_tmc_new.json'
with open(file, 'w') as f:
    json.dump(elbos, f)

file = 'results/timeseries_elbo_tmc.json'
with open(file, 'w') as f:
    json.dump(elbos_tmc, f)

file = 'results/timeseries_elbo_global.json'
with open(file, 'w') as f:
    json.dump(elbos_global, f)

# def Kdims_in_P(K):
#     #Individual Kdims in P
#
#     q_samples = []
#     logqs = []
#     logps = []
#     Ks = []
#     # Ks.append(Dim('K_ts_1', K))
#     Kdim = Dim('K', K)
#     q_samples.append(alan.Normal(0, 1/math.sqrt(N)).sample(sample_dims=(Kdim,), reparam=True))
#     logqs.append(alan.Normal(0, 1/math.sqrt(N)).log_prob(q_samples[-1]))
#     # for i in range(2,N+2):
#     #     q_samples.append(alan.Normal(q_samples[-1], 1/math.sqrt(N)).sample(sample_dims=(Kdim,), reparam=True))
#     #     logqs.append(alan.Normal(q_samples[-2], 1/math.sqrt(N)).log_prob(q_samples[-1]))
#
#     for i in range(2,N+1):
#         q_samples.append(alan.Normal(q_samples[-1], 1/math.sqrt(N)).sample(sample_dims=(Kdim,), reparam=True))
#         logqs.append(alan.Normal(q_samples[-2], 1/math.sqrt(N)).log_prob(q_samples[-1]))
#
#     # logqs.append(alan.Normal(q_samples[-1], 1/math.sqrt(N)).log_prob(data['obs']))
#
#
#     kdims = {'K_ts_{}'.format(i):Dim('K_ts_{}'.format(i), K) for i in range(N+1)}
#     # print(kdims)
#     q_samples[0] = q_samples[0].order(q_samples[0].dims)[(kdims['K_ts_0'],)]
#     logqs[0] = logqs[0].order(logqs[0].dims)[(kdims['K_ts_0'],)]
#
#     logps.append(alan.Normal(0, 1/math.sqrt(N)).log_prob(q_samples[0]))
#     for i in range(2,N+1):
#         q_samples[i-1] =  q_samples[i-1].order(q_samples[i-1].dims)[(kdims['K_ts_{}'.format(i-1)],)]
#         logqs[i-1] =  logqs[i-1].order(logqs[i-1].dims)[(kdims['K_ts_{}'.format(i-1)],)]
#         logps.append(alan.Normal(q_samples[i-2], 1/math.sqrt(N)).log_prob(q_samples[i-1]))
#
#     logps.append(alan.Normal(q_samples[-1], 1).log_prob(data['obs']))
#
#
#     tensors_p = logps
#     tensors_q = [-lq for lq in logqs]
#     factor = (tensors_p[0] + tensors_q[0])
#     # print(logps)
#     for i in range(1,N):
#
#         factor = (factor + tensors_p[i] + tensors_q[i])
#         factor = factor.exp().mean(kdims['K_ts_{}'.format(i-1)]).log()
#
#
#     factor = (tensors_p[-1] + factor).exp().mean(kdims['K_ts_{}'.format(N-1)]).log()
#
#     return (factor)
#
#
# #
# #
# # def Kdims_in_Q(K):
# #     #Individual Kdims in Q
# #
# #     q_samples = []
# #     logqs = []
# #     logps = []
# #     Ks = []
# #     # Ks.append(Dim('K_ts_1', K))
# #     Kdim = Dim('K', K)
# #     kdims = {'K_ts_{}'.format(i):Dim('K_ts_{}'.format(i), K) for i in range(N+1)}
# #     q_samples.append(alan.Normal(0, 1/math.sqrt(N)).sample(sample_dims=(kdims['K_ts_0'],), reparam=True))
# #     logqs.append(alan.Normal(0, 1/math.sqrt(N)).log_prob(q_samples[-1]))
# #     # for i in range(2,N+2):
# #     #     q_samples.append(alan.Normal(q_samples[-1], 1/math.sqrt(N)).sample(sample_dims=(Kdim,), reparam=True))
# #     #     logqs.append(alan.Normal(q_samples[-2], 1/math.sqrt(N)).log_prob(q_samples[-1]))
# #
# #     for i in range(2,N+1):
# #         q_samples.append(alan.Normal(q_samples[-1], 1/math.sqrt(N)).sample(sample_dims=(kdims['K_ts_{}'.format(i-1)],), reparam=True))
# #         logqs.append(alan.Normal(q_samples[-2], 1/math.sqrt(N)).log_prob(q_samples[-1]))
# #
# #     # logqs.append(alan.Normal(q_samples[-1], 1/math.sqrt(N)).log_prob(data['obs']))
# #
# #
# #     # kdims = {'K_ts_{}'.format(i):Dim('K_ts_{}'.format(i), K) for i in range(N+1)}
# #     # print(kdims)
# #     # q_samples[0] = q_samples[0].order(q_samples[0].dims)[(kdims['K_ts_0'],)]
# #     # logqs[0] = logqs[0].order(logqs[0].dims)[(kdims['K_ts_0'],)]
# #
# #     logps.append(alan.Normal(0, 1/math.sqrt(N)).log_prob(q_samples[0]))
# #     for i in range(2,N+1):
# #         # q_samples[i-1] =  q_samples[i-1].order(q_samples[i-1].dims)[(kdims['K_ts_{}'.format(i-1)],)]
# #         # logqs[i-1] =  logqs[i-1].order(logqs[i-1].dims)[(kdims['K_ts_{}'.format(i-1)],)]
# #         logps.append(alan.Normal(q_samples[i-2], 1/math.sqrt(N)).log_prob(q_samples[i-1]) - math.log(K))
# #
# #     logps.append(alan.Normal(q_samples[-1], 1/math.sqrt(N)).log_prob(data['obs']))
# #
# #     # print(logps)
# #     # print(logqs)
# #     tensors = [*logps, *[-lq for lq in logqs]]
# #     ## Convert tensors to Float64
# #     tensors = [x.to(dtype=t.float64) for x in tensors]
# #
# #
# #
# #     all_dims = unify_dims(tensors)
# #     # print(all_dims)
# #     Ks_to_sum    = all_dims
# #
# #     return reduce_Ks(tensors, Ks_to_sum)# - math.log(K**(N)))
#
#
#
#
# # def TMC(K):
# #     #Individual Kdims in P
# #
# #     q_samples = []
# #     logqs = []
# #     logps = []
# #     Ks = []
# #     # Ks.append(Dim('K_ts_1', K))
# #     Kdim = Dim('K', K)
# #     kdims = {'K_ts_{}'.format(i):Dim('K_ts_{}'.format(i), K) for i in range(N+1)}
# #     q_samples.append(alan.Normal(0, 1/math.sqrt(N)).sample(sample_dims=(kdims['K_ts_0'],), reparam=True))
# #     logqs.append(alan.Normal(0, 1/math.sqrt(N)).log_prob(q_samples[-1]))
# #     # for i in range(2,N+2):
# #     #     q_samples.append(alan.Normal(q_samples[-1], 1/math.sqrt(N)).sample(sample_dims=(Kdim,), reparam=True))
# #     #     logqs.append(alan.Normal(q_samples[-2], 1/math.sqrt(N)).log_prob(q_samples[-1]))
# #
# #     for i in range(2,N+1):
# #         # Ks = list(kdims.values())[:i-1]
# #         q_samples.append(alan.Normal(q_samples[-1], 1/math.sqrt(N)).sample(sample_dims=(kdims['K_ts_{}'.format(i-1)],), reparam=True))
# #         Ks = q_samples[-2].dims
# #
# #         idxs = [Categorical(t.ones(K)/K).sample(False, sample_dims=(kdims['K_ts_{}'.format(i-2)],)) for Kdim in Ks]
# #
# #         sample = q_samples[-1]
# #
# #         logqs.append(mean_dims(alan.Normal(q_samples[-2].order(*Ks)[idxs], 1/math.sqrt(N)).log_prob(sample).exp(), Ks).log() + math.log(K))
# #         # logqs.append(alan.Normal(q_samples[-2], 1/math.sqrt(N)).log_prob(q_samples[-1]))
# #
# #     # logqs.append(alan.Normal(q_samples[-1], 1/math.sqrt(N)).log_prob(data['obs']))
# #
# #
# #     # print(kdims)
# #     # q_samples[0] = q_samples[0].order(q_samples[0].dims)[(kdims['K_ts_0'],)]
# #     # logqs[0] = logqs[0].order(logqs[0].dims)[(kdims['K_ts_0'],)]
# #
# #     logps.append(alan.Normal(0, 1/math.sqrt(N)).log_prob(q_samples[0]))
# #     for i in range(2,N+1):
# #         # q_samples[i-1] =  q_samples[i-1].order(q_samples[i-1].dims)[(kdims['K_ts_{}'.format(i-1)],)]
# #         # logqs[i-1] =  logqs[i-1].order(logqs[i-1].dims)[(kdims['K_ts_{}'.format(i-1)],)]
# #         logps.append(alan.Normal(q_samples[i-2], 1/math.sqrt(N)).log_prob(q_samples[i-1]))
# #
# #     logps.append(alan.Normal(q_samples[-1], 1/math.sqrt(N)).log_prob(data['obs']))
# #
# #     tensors_p = logps
# #     tensors_q = [-lq for lq in logqs]
# #     factor = (tensors_p[0] + tensors_q[0])
# #     # print(logps)
# #     for i in range(1,N):
# #
# #         factor = (factor + tensors_p[i] + tensors_q[i])
# #         factor = factor.exp().mean(kdims['K_ts_{}'.format(i-1)]).log()
# #
# #
# #     factor = (tensors_p[-1] + factor).exp().mean(kdims['K_ts_{}'.format(N-1)]).log()
# #
# #     return (factor)
#
# def TMC(K):
#     #Individual Kdims in P
#
#     q_samples = []
#     logqs = []
#     logps = []
#     Ks = []
#     # Ks.append(Dim('K_ts_1', K))
#     Kdim = Dim('K', K)
#     kdims = {'K_ts_{}'.format(i):Dim('K_ts_{}'.format(i), K) for i in range(N+1)}
#     idxs = Categorical(t.ones(K)/K).sample(False, sample_dims=(kdims['K_ts_0'],))
#     q_samples.append(alan.Normal(0, 1/math.sqrt(N)).sample(sample_dims=(Kdim,), reparam=True).order((Kdim,))[idxs])
#     logqs.append(alan.Normal(0, 1/math.sqrt(N)).log_prob(q_samples[-1]))
#     # logqs.append(mean_dims(alan.Normal(0, 1/math.sqrt(N)).log_prob(q_samples[-1]).exp(), (Kdim,)).log())
#
#     for i in range(2,N+1):
#         Kdim = kdims['K_ts_{}'.format(i-1)]
#         Kdim_2 = kdims['K_ts_{}'.format(i-2)]
#         print(Kdim)
#         print(Kdim_2)
#         idxs = Categorical(t.ones(K)/K).sample(False, sample_dims=(Kdim,))
#         # print(idxs)
#         # print(q_samples[-1].order((Kdim,)))
#         # print(q_samples[-1].order((Kdim,))[idxs])
#         q_samples.append(alan.Normal(q_samples[-1].order((Kdim,))[idxs], 1/math.sqrt(N)).sample(sample_dims=(Kdim,), reparam=True).order((Kdim,))[idxs])
#         # logq   = mean_dims(dist.log_prob(sample).exp(), Ks).log()
#         print(alan.Normal(q_samples[-2], 1/math.sqrt(N)).log_prob(q_samples[-1]))
#         print(mean_dims(alan.Normal(q_samples[-2], 1/math.sqrt(N)).log_prob(q_samples[-1]), (Kdim,)))
#         logqs.append(mean_dims(alan.Normal(q_samples[-2], 1/math.sqrt(N)).log_prob(q_samples[-1]).exp(), (Kdim,)).log())
#         # print(logqs)
#
#
#
#     kdims = {'K_ts_{}'.format(i):Dim('K_ts_{}'.format(i), K) for i in range(N+1)}
#
#     q_samples[0] = q_samples[0].order(q_samples[0].dims)[(kdims['K_ts_0'],)]
#     logqs[0] = logqs[0].order(logqs[0].dims)[(kdims['K_ts_0'],)]
#
#     logps.append(alan.Normal(0, 1/math.sqrt(N)).log_prob(q_samples[0]))
#     for i in range(2,N+1):
#         q_samples[i-1] =  q_samples[i-1].order(q_samples[i-1].dims)[(kdims['K_ts_{}'.format(i-1)],)]
#         logqs[i-1] =  logqs[i-1].order(logqs[i-1].dims)[(kdims['K_ts_{}'.format(i-1)],)]
#         logps.append(alan.Normal(q_samples[i-2], 1/math.sqrt(N)).log_prob(q_samples[i-1]))
#
#     logps.append(alan.Normal(q_samples[-1], 1).log_prob(data['obs']))
#
#
#     tensors_p = logps
#     tensors_q = [-lq for lq in logqs]
#     factor = (tensors_p[0] + tensors_q[0])
#     # print(logps)
#     for i in range(1,N):
#
#         factor = (factor + tensors_p[i] + tensors_q[i])
#         factor = factor.exp().mean(kdims['K_ts_{}'.format(i-1)]).log()
#
#
#     factor = (tensors_p[-1] + factor).exp().mean(kdims['K_ts_{}'.format(N-1)]).log()
#
#     return (factor)
#
# model = alan.Model(P, Q, data)
# # # print(Kdims_in_P(1))
# # print(Kdims_in_P(3))
# # print(Kdims_in_P(10))
# # print(Kdims_in_P(30))
# # print(Kdims_in_P(300))
# # print(Kdims_in_P(1000))
# #
# #
# # # print('ELBO TMC')
# # # print(model.elbo_tmc(3))
# # # print(model.elbo_tmc(10))
# # # print(model.elbo_tmc(30))
# # # print(model.elbo_tmc(300))
# # # print(model.elbo_tmc(1000))
# #
# # # # print(TMC(1))
# # print(TMC(3))
# # print(TMC(10))
# # print(TMC(30))
# # print(TMC(300))
# # print(TMC(1000))
# Ks=[1,3,10,30,300, 1000]
# #Ks=[1]
# elbos = {k:0 for k in Ks}
# elbos_tmc = {k:0 for k in Ks}
# elbos_global = {k:0 for k in Ks}
# print(data)
# for k in Ks:
#
#
#     # elbos = []
#     num_runs = 100
#     for i in range(num_runs):
#         elbos[k] += (Kdims_in_P(k))/num_runs
#         elbos_tmc[k] += (model.elbo_tmc(k))/num_runs
#         elbos_global[k] += (model.elbo_global(k))/num_runs
# print(elbos)
# print(elbos_tmc)
# print(elbos_global)
#
#
# # model = alan.Model(P, Q, data)
# # # print('ELBO')
# # print(model.elbo(1))
# # print(model.elbo(10))
# # print(model.elbo(30))
# # print(model.elbo(100))
# # print(model.elbo(200))
# # print(model.elbo(500))
# #
# # print('ELBO TMC')
# # print(model.elbo_tmc(3))
# # print(model.elbo_tmc(10))
# # print(model.elbo_tmc(30))
# # print(model.elbo_tmc(100))
# # print(model.elbo_tmc(200))
# # print(model.elbo_tmc(500))
# #
# # print('ELBO GLOBAL')
# # print(model.elbo_tmc(3))
# # print(model.elbo_tmc(10))
# # print(model.elbo_tmc(30))
# # print(model.elbo_tmc(100))
# # print(model.elbo_tmc(200))
# # print(model.elbo_tmc(1000))
# # elbo_1 = []
# # elbo_3 = []
# # elbo_10 = []
# # elbo_30 = []
# #
# #
# # elbo_tmc_1 = []
# # elbo_tmc_3 = []
# # elbo_tmc_10 = []
# # elbo_tmc_30 = []
# #
# # num_runs = 1000
# # for i in range(num_runs):
# #     model = alan.Model(P, Q, data)
# #     # elbo_1.append(model.elbo(1))
# #     # elbo_3.append(model.elbo(3))
# #     # elbo_10.append(model.elbo(10))
# #     # elbo_30.append(model.elbo(30))
# #     # elbo_300.append(model.elbo(300))
# #
# #     elbo_1.append(Kdims_in_Q(1))
# #     elbo_3.append(Kdims_in_Q(3))
# #     elbo_10.append(Kdims_in_Q(10))
# #     # elbo_30.append(Kdims_in_Q(30))
# #
# #     elbo_tmc_1.append(model.elbo_tmc(1))
# #     elbo_tmc_3.append(model.elbo_tmc(3))
# #     elbo_tmc_10.append(model.elbo_tmc(10))
# #     # elbo_tmc_30.append(model.elbo_tmc(30))
# #
# #
# # print(sum(elbo_1)/num_runs)
# # print(sum(elbo_3)/num_runs)
# # print(sum(elbo_10)/num_runs)
# # # print(sum(elbo_30)/num_runs)
# #
# #
# # print(sum(elbo_tmc_1)/num_runs)
# # print(sum(elbo_tmc_3)/num_runs)
# # print(sum(elbo_tmc_10)/num_runs)
# # print(sum(elbo_tmc_30)/num_runs)
