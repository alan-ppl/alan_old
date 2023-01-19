import torch as t
import torch.nn as nn
import alan
import numpy as np
from alan.experiment_utils import seed_torch


seed_torch(0)


N = 25
def P(tr):
    tr.sample('ts_1', alan.Normal(0, 1/np.sqrt(N)))
    # print(tr['ts_1'])
    for i in range(2,N+1):
        tr.sample('ts_{}'.format(i), alan.Normal(tr['ts_{}'.format(i-1)], 1/np.sqrt(N)))
        # print(tr['ts_{}'.format(i)])
    tr.sample('obs', alan.Normal(tr['ts_{}'.format(N)], 1))

def Q(tr):
    tr.sample('ts_1', alan.Normal(0, 1/np.sqrt(N)))
    for i in range(2,N+1):
        tr.sample('ts_{}'.format(i), alan.Normal(tr['ts_{}'.format(i-1)], 1/np.sqrt(N)))


data = alan.sample(P, varnames=('obs',))


model = alan.Model(P, Q, data)
print(model.elbo(1))
print(model.elbo(3))
print(model.elbo(10))
print(model.elbo(30))
print(model.elbo(100))
print(model.elbo(200))
print(model.elbo(500))

print(model.elbo_tmc(1))
print(model.elbo_tmc(3))
print(model.elbo_tmc(10))
print(model.elbo_tmc(30))
print(model.elbo_tmc(100))
print(model.elbo_tmc(200))
print(model.elbo_tmc(500))
#
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
