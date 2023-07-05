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

import alan.postproc as pp

z_mean = 20
z_var = 10
obs_var = 0.1
def P(tr):
    tr('z', alan.Normal(z_mean,z_var))
    tr('obs', alan.Normal(tr['z'], obs_var), plates='plate_1')

class Q_ML(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.Nz = alan.AMMP_ISNormal()


    def forward(self, tr):
        tr('z',   self.Nz())


class Q(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.mu_z = nn.Parameter(t.zeros(()))
        self.log_s_z = nn.Parameter(t.zeros(()))


    def forward(self, tr):
        tr('z',   alan.Normal(self.mu_z, self.log_s_z.exp()))

all_data = alan.Model(P).sample_prior(varnames='obs', platesizes = {'plate_1':2})
data = {}
data['obs'], _ = t.split(all_data['obs'].clone(), [1,1], -1)
# True var:
var = 1/((1/obs_var)+(1/z_var))

#True mean
mean = var*(z_mean/z_var + data['obs'].item()/obs_var)

print(f'True mean: {mean}')
print(f'True var: {var}')




K = 100
T = 2000
ml_lrs = [0.1]
# ml_lrs = [3]
vi_lrs = [0.5]
ml_colours = ['#ffffb2','#fecc5c','#fd8d3c','#f03b20'][::-1]
vi_colours = ['#edf8fb','#b2e2e2','#66c2a4','#2ca25f'][::-1]
fig, ax = plt.subplots(4,1, figsize=(5.5, 8.0))
for j in range(len(ml_lrs)):
    lr = vi_lrs[j]
    means = []
    scales = []
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
        means.append(pp.mean(sample.weights())['z'].item())   
        scales.append(pp.std(sample.weights())['z'].item())
        elbo = sample.elbo()
        elbos.append(elbo.item())
        start = time.time()
        (-elbo).backward()
        opt.step()
        times.append(time.time() - start)


        sample = cond_model.sample_same(K, reparam=False, device=t.device('cpu'))
        pred_lls.append(cond_model.predictive_ll(sample, N = 10, data_all=all_data)['obs'])


        if i % 500 == 0:
            print(f'Elbo: {elbo.item()}')        


    elbos = np.expand_dims(np.array(elbos), axis=0)
    pred_lls = np.expand_dims(np.array(pred_lls), axis=0)

    ax[0].plot(np.cumsum(times), means, color=vi_colours[j], label=f'Vi lr: {lr}')
    ax[0].axhline(mean)
    ax[1].plot(np.cumsum(times), scales, color=vi_colours[j])
    ax[1].axhline(var)
    ax[2].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=vi_colours[j])
    ax[3].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=vi_colours[j])


    lr = ml_lrs[j]
    means = []
    scales = []
    elbos = []
    times = []
    pred_lls = []
    t.manual_seed(0)
    q = Q_ML()
    m1 = alan.Model(P, q).condition(data=data)

    for i in range(T):
        sample = m1.sample_same(K, reparam=False)
        means.append(pp.mean(sample.weights())['z'].item())   
        scales.append(pp.std(sample.weights())['z'].item())
        if i % 500 == 0:
            # print(q.Nz.mean2conv(*q.Nz.named_means))
            print(f'Elbo: {sample.elbo().item()}')   
        elbos.append(sample.elbo().item()) 
        start = time.time()    
        m1.ammpis_update(lr, sample)
        times.append(time.time() - start)


        sample = m1.sample_same(K, reparam=False, device=t.device('cpu'))
        pred_lls.append(m1.predictive_ll(sample, N = 10, data_all=all_data)['obs'])


    elbos = np.expand_dims(np.array(elbos), axis=0)
    pred_lls = np.expand_dims(np.array(pred_lls), axis=0)

    ax[0].plot(np.cumsum(times), means, color=ml_colours[j], label=f'AMMP-IS lr: {lr}')
    ax[0].axhline(mean)
    ax[1].plot(np.cumsum(times), scales, color=ml_colours[j])
    ax[1].axhline(var)
    ax[2].plot(np.cumsum(times), elbos.squeeze(0), color=ml_colours[j])
    ax[3].plot(np.cumsum(times), pred_lls.squeeze(0), color=ml_colours[j])


ax[0].set_title(f'K: {K}')

ax[0].set_ylabel('Mean')
ax[0].set_ylim(30,40)

ax[1].set_ylabel('Scale')
ax[1].set_ylim(-0.25,0.35)

ax[2].set_ylabel('ELBO')
ax[2].set_ylim(-20,10)
ax[3].set_ylabel('Predictive LL')
ax[3].set_ylim(-20,10)

ax[3].set_xlabel('Time')

ax[0].legend(loc='upper right')



plt.savefig('chart.png')