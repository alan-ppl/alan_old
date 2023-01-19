import torch as t
import torch.nn as nn
import alan
import math
from alan.experiment_utils import seed_torch
from functorch.dim import dims, Dim
from alan.utils import *

seed_torch(0)


N = 50
def P(tr):
    tr.sample('ts_1', alan.Normal(0, 1/math.sqrt(N)))
    # print(tr['ts_1'])
    for i in range(2,N+1):
        tr.sample('ts_{}'.format(i), alan.Normal(tr['ts_{}'.format(i-1)], 1/math.sqrt(N)))
        # print(tr['ts_{}'.format(i)])
    tr.sample('obs', alan.Normal(tr['ts_{}'.format(N)], 1))

def Q(tr):
    tr.sample('ts_1', alan.Normal(0, 1/math.sqrt(N)))
    for i in range(2,N+1):
        tr.sample('ts_{}'.format(i), alan.Normal(tr['ts_{}'.format(i-1)], 1/math.sqrt(N)))


data = alan.sample(P, varnames=('obs',))


K=1000
q_samples = []
logqs = []
logps = []
Ks = []
# Ks.append(Dim('K_ts_1', K))
Kdim = Dim('K', K)
q_samples.append(alan.Normal(0, 1/math.sqrt(N)).sample(sample_dims=(Kdim,), reparam=True))
logqs.append(alan.Normal(0, 1/math.sqrt(N)).log_prob(q_samples[-1]))
for i in range(2,N+2):
    q_samples.append(alan.Normal(q_samples[-1], 1/math.sqrt(N)).sample(sample_dims=(Kdim,), reparam=True))
    logqs.append(alan.Normal(q_samples[-2], 1/math.sqrt(N)).log_prob(q_samples[-1]))

#logqs.append(alan.Normal(q_samples[-1], 1/math.sqrt(N)).log_prob(data['obs']))
# print(q_samples)


kdims = {'K_ts_{}'.format(i):Dim('K_ts_{}'.format(i), K) for i in range(N+1)}
# print(kdims)
q_samples[0] = q_samples[0].order(q_samples[0].dims)[(kdims['K_ts_0'],)]
logqs[0] = logqs[0].order(logqs[0].dims)[(kdims['K_ts_0'],)]

logps.append(alan.Normal(0, 1/math.sqrt(N)).log_prob(q_samples[0]))
for i in range(2,N+2):
    q_samples[i-1] =  q_samples[i-1].order(q_samples[i-1].dims)[(kdims['K_ts_{}'.format(i-1)],)]
    logqs[i-1] =  logqs[i-1].order(logqs[i-1].dims)[(kdims['K_ts_{}'.format(i-1)],)]
    logps.append(alan.Normal(q_samples[i-2], 1/math.sqrt(N)).log_prob(q_samples[i-1]))

logps.append(alan.Normal(q_samples[-1], 1/math.sqrt(N)).log_prob(data['obs']))
# print(logqs)
# print(logps)

tensors = [*logps, *[-lq for lq in logqs]]
## Convert tensors to Float64
tensors = [x.to(dtype=t.float64) for x in tensors]



all_dims = unify_dims(tensors)
# print(all_dims)
Ks_to_sum    = all_dims

print(reduce_Ks(tensors, Ks_to_sum) - math.log(K**(N+1)))

# Ks=[1,3,10,30,300]# 1000]
# #Ks=[1]
# elbos = {k:0 for k in Ks}
# elbos_tmc = {k:0 for k in Ks}
model = alan.Model(P, Q, data)
# print(data)
# for k in Ks:
#
#
#     # elbos = []
#     num_runs = 100
#     for i in range(num_runs):
#         samples = model._sample(K=k, data=None, covariates=None, reparam=True)
#         samples_tmc = model._sample_tmc(K=k, data=None, covariates=None, reparam=True)
#
#         # print(samples.trp.logp)
#         # print(samples.trp.logq)
#         #
#         # print(samples_tmc.trp.logp)
#         # print(samples_tmc.trp.logq)
#
#         # samples_tmc.trp.logq = {key: value + math.log(k) for key,value in samples_tmc.trp.logq.items()}# if key != 'obs'}
#         #
#         # samples.trp.logp = {key: value - math.log(k) for key,value in samples.trp.logp.items()}# if key != 'obs'}
#         # print(math.log(float(k**(N))))
#         # print(math.log(k))
#         elbos[k] += (samples.tensor_product() - math.log(float(k**(N))))/num_runs
#         elbos_tmc[k] += (samples_tmc.tensor_product() - math.log(float(k**N)))/num_runs
#
#         # elbos.append(model.elbo(K) - np.log(K**N))
#
#
# print(elbos)
# print(elbos_tmc)




# print('ELBO')
# print(model.elbo(3))
# print(model.elbo(10))
# print(model.elbo(30))
# print(model.elbo(100))
# print(model.elbo(200))
# print(model.elbo(500))
#
# print('ELBO TMC')
# print(model.elbo_tmc(3))
# print(model.elbo_tmc(10))
# print(model.elbo_tmc(30))
# print(model.elbo_tmc(100))
# print(model.elbo_tmc(200))
# print(model.elbo_tmc(500))
#
# print('ELBO GLOBAL')
# print(model.elbo_tmc(3))
# print(model.elbo_tmc(10))
# print(model.elbo_tmc(30))
# print(model.elbo_tmc(100))
# print(model.elbo_tmc(200))
print(model.elbo_tmc(1000))
# elbo_1 = []
# elbo_3 = []
# elbo_10 = []
# elbo_30 = []
#
# elbo_tmc_1 = []
# elbo_tmc_3 = []
# elbo_tmc_10 = []
# elbo_tmc_30 = []
# num_runs = 25
# for i in range(num_runs):
#     model = alan.Model(P, Q, data)
#     elbo_1.append(model.elbo(1))
#     elbo_3.append(model.elbo(3))
#     elbo_10.append(model.elbo(10))
#     elbo_30.append(model.elbo(30))
#
#     elbo_tmc_1.append(model.elbo_tmc(1))
#     elbo_tmc_3.append(model.elbo_tmc(3))
#     elbo_tmc_10.append(model.elbo_tmc(10))
#     elbo_tmc_30.append(model.elbo_tmc(30))
#
# print(sum(elbo_1)/num_runs)
# print(sum(elbo_3)/num_runs)
# print(sum(elbo_10)/num_runs)
# print(sum(elbo_30)/num_runs)
#
# print(sum(elbo_tmc_1)/num_runs)
# print(sum(elbo_tmc_3)/num_runs)
# print(sum(elbo_tmc_10)/num_runs)
# print(sum(elbo_tmc_30)/num_runs)
