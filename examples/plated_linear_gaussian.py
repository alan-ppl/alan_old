import torch as t
import torch.nn as nn
import alan
import numpy as np
t.manual_seed(0)

J = 2
M = 3
N = 4
platesizes = {'plate_1': J, 'plate_2': M, 'plate_3': N}
def P(tr):

    tr.sample('a',   alan.Normal(t.zeros(()), 1))

    tr.sample('b',   alan.Normal(tr['a'], 1))

    tr.sample('c',   alan.Normal(tr['b'], 1), plates='plate_1')

    tr.sample('d',   alan.Normal(tr['c'], 1), plates='plate_2')

    tr.sample('obs', alan.Normal(tr['d'], 1), plates='plate_3')


# def Q(tr):
#
#     tr.sample('a',   alan.Normal(t.zeros(()), 1))
#
#     tr.sample('b',   alan.Normal(tr['a'], 1))
#
#     tr.sample('c',   alan.Normal(tr['b'], 1), plates='plate_1')
#
#     tr.sample('d',   alan.Normal(tr['c'], 1), plates='plate_2')
#
#     # tr.sample('obs', alan.Normal(tr['d'], 1), plates='plate_3')

# class Q(alan.QModule):
#     def __init__(self):
#         super().__init__()
#         self.m_a = nn.Parameter(t.zeros(()))
#         self.w_b = nn.Parameter(t.zeros(()))
#         self.b_b = nn.Parameter(t.zeros(()))
#
#         self.w_c = nn.Parameter(t.zeros((J,), names=('plate_1',)))
#         self.b_c = nn.Parameter(t.zeros((J,), names=('plate_1',)))
#
#         self.w_d = nn.Parameter(t.zeros((M, J), names=('plate_2','plate_1')))
#         self.b_d = nn.Parameter(t.zeros((M, J), names=('plate_2','plate_1')))
#
#         self.log_s_a = nn.Parameter(t.zeros(()))
#         self.log_s_b = nn.Parameter(t.zeros(()))
#         self.log_s_c = nn.Parameter(t.zeros((J,), names=('plate_1',)))
#         self.log_s_d = nn.Parameter(t.zeros((M,J), names=('plate_2','plate_1')))
#
#
#     def forward(self, tr):
#
#         tr.sample('a', alan.Normal(self.m_a, self.log_s_a.exp()))
#
#         mean_b = self.w_b * tr['a'] + self.b_b
#         tr.sample('b', alan.Normal(mean_b, self.log_s_b.exp()))
#
#         mean_c = self.w_c * tr['b'] + self.b_c
#         tr.sample('c', alan.Normal(mean_c, self.log_s_c.exp()))
#
#         mean_d = self.w_d * tr['c'] + self.b_d
#         tr.sample('d', alan.Normal(mean_d, self.log_s_d.exp()))

class Q(alan.QModule):
    def __init__(self):
        super().__init__()
        self.m_a = nn.Parameter(t.zeros(()))
        self.w_b = nn.Parameter(t.zeros(()))


        self.w_c = nn.Parameter(t.zeros((J,), names=('plate_1',)))


        self.w_d = nn.Parameter(t.zeros((M, J), names=('plate_2','plate_1')))


        self.log_s_a = nn.Parameter(t.zeros(()))
        self.log_s_b = nn.Parameter(t.zeros(()))
        self.log_s_c = nn.Parameter(t.zeros((J,), names=('plate_1',)))
        self.log_s_d = nn.Parameter(t.zeros((M,J), names=('plate_2','plate_1')))


    def forward(self, tr):

        tr.sample('a', alan.Normal(self.m_a, self.log_s_a.exp()))

        # mean_b = self.w_b * tr['a'] + self.b_b
        tr.sample('b', alan.Normal(self.w_b, self.log_s_b.exp()))

        # mean_c = self.w_c * tr['b'] + self.b_c
        tr.sample('c', alan.Normal(self.w_c, self.log_s_c.exp()))

        # mean_d = self.w_d * tr['c'] + self.b_d
        tr.sample('d', alan.Normal(self.w_d, self.log_s_d.exp()))

data = alan.sample(P, platesizes=platesizes, varnames=('obs',))
test_data = alan.sample(P, platesizes=platesizes, varnames=('obs',))

all_data = {'obs': t.cat([data['obs'].rename(None),test_data['obs'].rename(None)], -3).rename('plate_3','plate_2','plate_1')}



global_k_predlls = []
tmc_predlls = []
tmc_new_predlls = []
mp_predlls = []
for i in range(100):
    num_iters = 1
    K=100
    t.manual_seed(0)
    model = alan.Model(P, Q(), {'obs': data['obs']})


    tmc_new_predlls.append(model.predictive_ll(K = K, N = 1000, data_all=all_data, sample_method='tmc_new')['obs'])

    t.manual_seed(0)
    model = alan.Model(P, Q(), {'obs': data['obs']})


    global_k_predlls.append(model.predictive_ll(K = K, N = 1000, data_all=all_data, sample_method='global_k')['obs'])

    t.manual_seed(0)
    model = alan.Model(P, Q(), {'obs': data['obs']})
    mp_predlls.append(model.predictive_ll(K = K, N = 1000, data_all=all_data, sample_method='MP')['obs'])

    t.manual_seed(0)
    model = alan.Model(P, Q(), {'obs': data['obs']})
    tmc_predlls.append(model.predictive_ll(K = K, N = 1000, data_all=all_data, sample_method='tmc')['obs'])


print(np.mean(tmc_predlls))
print(np.mean(tmc_new_predlls))
print(np.mean(global_k_predlls))
print(np.mean(mp_predlls))
# # Specify a path
# PATH = "state_dict_model.pt"
#
# # Save
# t.save(model.state_dict(), PATH)
