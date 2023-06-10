import matplotlib
matplotlib.use('TkAgg')

import torch as t
import torch.nn as nn
import alan
import matplotlib.pyplot as plt
import numpy as np
t.manual_seed(0)
from alan.experiment_utils import n_mean
import time

from alan.postproc import mean, var


z_mean = 20
z_var = 10
obs_var = 0.001
def P(tr):
    tr('z', alan.Normal(z_mean,z_var))
    tr('z_scale', alan.Normal(1, 1))
    tr('obs', alan.Normal(tr['z'], tr['z_scale'].exp()), plates='plate_1')

class Q_ML(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.Nz = alan.ML2Normal()
        self.Nz_scale = alan.ML2Normal()

    def forward(self, tr):
        tr('z',   self.Nz())
        tr('z_scale', self.Nz_scale())


class Q(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.mu_z = nn.Parameter(t.zeros(()))
        self.log_s_z = nn.Parameter(t.zeros(()))
        self.mu_z_scale = nn.Parameter(t.zeros(()))
        self.log_s_z_scale = nn.Parameter(t.zeros(()))

    def forward(self, tr):
        tr('z',   alan.Normal(self.mu_z, self.log_s_z.exp()))
        tr('z_scale',   alan.Normal(self.mu_z_scale, self.log_s_z_scale.exp()))


all_data = alan.Model(P).sample_prior(platesizes = {'plate_1':2})
data = {}
data['obs'], _ = t.split(all_data['obs'].clone(), [1,1], -1)

# # True var:
# var = 1/((1/obs_var)+(1/z_var))

# #True mean
# mean = var*(z_mean/z_var + data['obs'].item()/obs_var)

# True var
z_scale_mean = all_data['z_scale']

# True mean
z_mean = all_data['z']


all_data = {'obs': all_data['obs']}
print(f'True mean: {z_mean}')
print(f'True var: {z_scale_mean}')

# samp = mo
# print(print(mean(model.weights(10000))))

K = 300
T = 1000
ml_lrs = [0.99, 0.5]
vi_lrs = [0.01, 0.1]
ml_colours = ['#ffffb2','#fecc5c','#fd8d3c','#f03b20'][::-1]
vi_colours = ['#edf8fb','#b2e2e2','#66c2a4','#2ca25f'][::-1]
fig, ax = plt.subplots(6,1, figsize=(5.5, 8.0))
for j in range(len(ml_lrs)):
    lr = ml_lrs[j]
    z_means = []
    z_scales = []
    z_scale_means = []
    z_scale_scales = []
    elbos = []
    times = []
    pred_lls = []
    t.manual_seed(0)
    q = Q_ML()
    m1 = alan.Model(P, q).condition(data=data)

    # samp = m1.sample_same(K, reparam=False)
    # mean_post = mean(samp.weights(1000))
    # var_post

    for i in range(T):
        sample = m1.sample_same(K, reparam=False)
        z_means.append(q.Nz.mean2conv(*q.Nz.named_means)['loc'].item())   
        z_scales.append(q.Nz.mean2conv(*q.Nz.named_means)['scale'].item()) 
        z_scale_means.append(q.Nz.mean2conv(*q.Nz_scale.named_means)['loc'].item())
        z_scale_scales.append(q.Nz.mean2conv(*q.Nz_scale.named_means)['scale'].item()) 
        elbos.append(sample.elbo().item()) 
        if i % 500 == 0:
            # print(q.Nz.mean2conv(*q.Nz.named_means))
            print(f'Elbo: {elbos[-1]}')   
        
        start = time.time()    
        m1.update(lr, sample)
        times.append(time.time() - start)


        sample = m1.sample_same(K, reparam=False, device=t.device('cpu'))
        pred_lls.append(m1.predictive_ll(sample, N = 10, data_all=all_data)['obs'])


    elbos = np.expand_dims(np.array(elbos), axis=0)
    pred_lls = np.expand_dims(np.array(pred_lls), axis=0)

    ax[0].plot(np.cumsum(times), z_means, color=ml_colours[j], label=f'ML lr: {lr}')
    ax[0].axhline(z_mean)
    ax[1].plot(np.cumsum(times), z_scales, color=ml_colours[j])
    ax[2].axhline(z_scale_mean)
    ax[2].plot(np.cumsum(times), z_scale_means, color=ml_colours[j])
    ax[3].plot(np.cumsum(times), z_scale_scales, color=ml_colours[j])
    ax[4].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=ml_colours[j])
    ax[5].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=ml_colours[j])

    # try:
    lr = vi_lrs[j]
    z_means = []
    z_scales = []
    z_scale_means = []
    z_scale_scales = []
    elbos = []
    times = []
    pred_lls = []
    t.manual_seed(0)
    q = Q()
    cond_model = alan.Model(P, q).condition(data=data)
    opt = t.optim.Adam(cond_model.parameters(), lr=lr)
    for i in range(T):
        opt.zero_grad()
        sample = cond_model.sample_same(K, reparam=True)
        z_means.append(q.mu_z.item())   
        z_scales.append(q.log_s_z.exp().item())
        z_scale_means.append(q.mu_z_scale.item())
        z_scale_scales.append(q.log_s_z_scale.exp().item())
        elbo = sample.elbo()
        elbos.append(elbo.item())
        
        start = time.time()
        (-elbo).backward()
        opt.step()
        times.append(time.time() - start)


        sample = cond_model.sample_same(K, reparam=False, device=t.device('cpu'))
        pred_lls.append(cond_model.predictive_ll(sample, N = 10, data_all=all_data)['obs'])


        if i % 500 == 0:
            print(f'Elbo: {elbos[-1]}')        


    elbos = np.expand_dims(np.array(elbos), axis=0)
    pred_lls = np.expand_dims(np.array(pred_lls), axis=0)

    ax[0].plot(np.cumsum(times), z_means, color=vi_colours[j], label=f'VI lr: {lr}')
    ax[0].axhline(z_mean)
    ax[1].plot(np.cumsum(times), z_scales, color=vi_colours[j])
    ax[2].axhline(z_scale_mean)
    ax[2].plot(np.cumsum(times), z_scale_means, color=vi_colours[j])
    ax[3].plot(np.cumsum(times), z_scale_scales, color=vi_colours[j])
    ax[4].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=vi_colours[j])
    ax[5].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=vi_colours[j])
    # except:
    #     print(f'VI failing for lr: {lr}')

ax[0].set_ylabel('Mean')

ax[1].set_ylabel('Mean Scale')


ax[2].set_ylabel('Scale Mean')

ax[3].set_ylabel('Scale var')


ax[4].set_ylabel('ELBO')

ax[5].set_ylabel('Predictive LL')

ax[5].set_xlabel('Time')

# ax[0].set_ylim(0,50)
# # ax[0].set_xlim(-0.001,0.5)

# ax[1].set_ylim(0,50)
# # ax[1].set_xlim(-0.001,0.5)

ax[1].set_ylim(0,100)

# ax[3].set_ylim(-20,-18)

ax[0].legend(loc='upper right')



plt.savefig('chart_variance.png')