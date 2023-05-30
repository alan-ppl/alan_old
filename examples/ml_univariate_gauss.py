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



z_mean = 20
z_var = 10
obs_var = 0.001
def P(tr):
    tr('z', alan.Normal(z_mean,z_var))
    tr('obs', alan.Normal(tr['z'], obs_var))

class Q_ML(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.Nz = alan.MLNormal()


    def forward(self, tr):
        tr('z',   self.Nz())


class Q(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.mu_z = nn.Parameter(t.zeros(()))
        self.log_s_z = nn.Parameter(t.zeros(()))


    def forward(self, tr):
        tr('z',   alan.Normal(self.mu_z, self.log_s_z.exp()))

data = alan.Model(P).sample_prior(varnames='obs')

# True var:
var = 1/((1/obs_var)+(1/z_var))

#True mean
mean = var*(z_mean/z_var + data['obs'].item()/obs_var)

print(f'True mean: {mean}')
print(f'True var: {var}')




K = 100
T = 2500
ml_lrs = [1.5,0.9, 0.3]
vi_lrs = [2.5, 1, 0.3]
ml_colours = ['#ffffb2','#fecc5c','#fd8d3c','#f03b20'][::-1]
vi_colours = ['#edf8fb','#b2e2e2','#66c2a4','#2ca25f'][::-1]
fig, ax = plt.subplots(3,1, figsize=(5.5, 8.0))
for j in range(len(ml_lrs)):
    lr = ml_lrs[j]
    means = []
    scales = []
    elbos = []
    times = []
    t.manual_seed(0)
    q = Q_ML()
    m1 = alan.Model(P, q).condition(data=data)

    for i in range(T):
        sample = m1.sample_same(K, reparam=False)
        means.append(q.Nz.mean2conv(*q.Nz.named_means)['loc'].item())   
        scales.append(q.Nz.mean2conv(*q.Nz.named_means)['scale'].item())
        if i % 500 == 0:
            # print(q.Nz.mean2conv(*q.Nz.named_means))
            print(f'Elbo: {sample.elbo().item()}')   
        elbos.append(sample.elbo().item()) 
        start = time.time()    
        m1.update(lr, sample)
        times.append(time.time() - start)


    elbos = np.expand_dims(np.array(elbos), axis=0)
    

    ax[0].plot(np.cumsum(times), means, color=ml_colours[j], label=f'ML lr: {lr}')
    ax[0].axhline(mean)
    ax[1].plot(np.cumsum(times), scales, color=ml_colours[j])
    ax[1].axhline(var)
    ax[2].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=ml_colours[j])


    lr = vi_lrs[j]
    means = []
    scales = []
    elbos = []
    times = []
    t.manual_seed(0)
    q = Q()
    cond_model = alan.Model(P, q).condition(data=data)
    opt = t.optim.Adam(cond_model.parameters(), lr=lr)
    for i in range(T):
        opt.zero_grad()
        sample = cond_model.sample_same(K, reparam=True)
        means.append(q.mu_z.item())   
        scales.append(q.log_s_z.exp().item())
        elbo = sample.elbo()
        elbos.append(elbo.item())
        start = time.time()
        (-elbo).backward()
        opt.step()
        times.append(time.time() - start)
    
        if i % 500 == 0:
            print(f'Elbo: {elbo.item()}')        


    elbos = np.expand_dims(np.array(elbos), axis=0)
    

    ax[0].plot(np.cumsum(times), means, color=vi_colours[j], label=f'Vi lr: {lr}')
    ax[0].axhline(mean)
    ax[1].plot(np.cumsum(times), scales, color=vi_colours[j])
    ax[1].axhline(var)
    ax[2].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=vi_colours[j])





ax[0].set_ylabel('Mean')
# ax[0].set_ylim(30,50)
# ax[0].set_xlim(-0.001,0.5)
ax[1].set_ylabel('Scale')
# ax[1].set_ylim(0,0.4)
# ax[1].set_xlim(-0.001,0.5)
ax[2].set_ylabel('ELBO')
# ax[2].set_ylim(-6,-3)


ax[2].set_xlabel('Time')

ax[0].legend(loc='upper right')



plt.savefig('chart.png')