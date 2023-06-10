import torch as t
import torch.nn as nn

import alan
import alan.postproc as pp


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import time


from alan.experiment_utils import seed_torch, n_mean

from movielens.movielens import generate_model as generate_ML
from movielens.movielens_VI import generate_model as generate_VI

ml_colours = ['#ffffb2','#fecc5c','#fd8d3c','#f03b20'][::-1]
vi_colours = ['#edf8fb','#b2e2e2','#66c2a4','#2ca25f'][::-1]
seed_torch(0)

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

M=300
N=5

P, Q_ml, data, covariates, all_data, all_covariates, sizes = generate_ML(N,M, device, 2, 0, True)
P, Q_vi, _, _, _, _, _ = generate_VI(N,M, device, 2, 0, True)

# # True psi_z
# z_scale_mean = data['mu_z'][0]

# # True mean
# z_mean = data['psi_z'][0]

# True posterior psi_z

with open('psi_z_posterior_mean.pkl', 'rb') as f:
    z_scale_mean = pickle.load(f)


# True posterior mean
with open('mu_z_posterior_mean.pkl', 'rb') as f:
    z_mean = pickle.load(f)


data = {'obs':data.pop('obs')}

K = 10
T = 200
ml_lrs = [0.15,0.1]
vi_lrs = [0.15,0.1,0.05]

fig, ax = plt.subplots(6,1, figsize=(7, 8.0))
for j in range(len(ml_lrs)):
    lr = ml_lrs[j]
    print(f'ML: {lr}')
    z_means = []
    z_scales = []
    z_scale_means = []
    z_scale_scales = []
    elbos = []
    times = []
    pred_lls = []
    seed_torch(0)
    q = Q_ml()
    m1 = alan.Model(P, q).condition(data=data)

    # samp = m1.sample_same(K, reparam=False)
    # mean_post = mean(samp.weights(1000))
    # var_post

    for i in range(T):
        sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)
        z_means.append(q.mu.mean2conv(*q.mu.named_means)['loc'][0].item())   
        z_scales.append(q.mu.mean2conv(*q.mu.named_means)['scale'][0].item()) 
        z_scale_means.append(q.psi_z.mean2conv(*q.psi_z.named_means)['loc'][0].item())
        z_scale_scales.append(q.psi_z.mean2conv(*q.psi_z.named_means)['scale'][0].item()) 
        elbos.append(sample.elbo().item()) 
        if i % 100 == 0:
            # print(q.Nz.mean2conv(*q.Nz.named_means))
            print(f'Elbo: {elbos[-1]}')   
        
        start = time.time()    
        m1.update(lr, sample)
        times.append(time.time() - start)


        sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)
        pred_lls.append(m1.predictive_ll(sample, N = 10,inputs_all=all_covariates, data_all=all_data)['obs'])


    elbos = np.expand_dims(np.array(elbos), axis=0)
    pred_lls = np.expand_dims(np.array(pred_lls), axis=0)

    ax[0].plot(np.cumsum(times), z_means, color=ml_colours[j], label=f'ML lr: {lr}')
    ax[0].axhline(z_mean)
    ax[1].plot(np.cumsum(times), z_scales, color=ml_colours[j])
    ax[2].axhline(z_scale_mean)
    ax[2].plot(np.cumsum(times), z_scale_means, color=ml_colours[j])
    ax[3].plot(np.cumsum(times), z_scale_scales, color=ml_colours[j])
    # ax[4].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=ml_colours[j])
    # ax[5].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=ml_colours[j])
    ax[4].plot(np.cumsum(times), elbos.squeeze(0), color=ml_colours[j])
    ax[5].plot(np.cumsum(times), pred_lls.squeeze(0), color=ml_colours[j])

for j in range(len(vi_lrs)):
    lr = vi_lrs[j]
    print(f'VI: {lr}')
    z_means = []
    z_scales = []
    z_scale_means = []
    z_scale_scales = []
    elbos = []
    times = []
    pred_lls = []
    seed_torch(0)
    q = Q_vi()
    cond_model = alan.Model(P, q).condition(data=data)
    opt = t.optim.Adam(cond_model.parameters(), lr=lr)
    for i in range(T):
        opt.zero_grad()
        sample = cond_model.sample_same(K, reparam=True, inputs=covariates, device=device)
        z_means.append(q.m_mu_z[0].item())   
        z_scales.append(q.log_theta_mu_z[0].exp().item())
        z_scale_means.append(q.m_psi_z[0].item())
        z_scale_scales.append(q.log_theta_psi_z[0].exp().item())
        elbo = sample.elbo()
        elbos.append(elbo.item())
        
        start = time.time()
        (-elbo).backward()
        opt.step()
        times.append(time.time() - start)


        sample = cond_model.sample_same(K, reparam=False, inputs=covariates, device=device)
        pred_lls.append(cond_model.predictive_ll(sample, N = 10, inputs_all=all_covariates, data_all=all_data)['obs'])


        if i % 100 == 0:
            print(f'Elbo: {elbos[-1]}')        


    elbos = np.expand_dims(np.array(elbos), axis=0)
    pred_lls = np.expand_dims(np.array(pred_lls), axis=0)

    ax[0].plot(np.cumsum(times), z_means, color=vi_colours[j], label=f'VI lr: {lr}')
    ax[0].axhline(z_mean)
    ax[1].plot(np.cumsum(times), z_scales, color=vi_colours[j])
    ax[2].axhline(z_scale_mean)
    ax[2].plot(np.cumsum(times), z_scale_means, color=vi_colours[j])
    ax[3].plot(np.cumsum(times), z_scale_scales, color=vi_colours[j])
    # ax[4].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=vi_colours[j])
    # ax[5].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=vi_colours[j])
    ax[4].plot(np.cumsum(times), elbos.squeeze(0), color=vi_colours[j])
    ax[5].plot(np.cumsum(times), pred_lls.squeeze(0), color=vi_colours[j])

ax[0].set_ylabel('mu_z')

ax[1].set_ylabel('mu_z Scale')


ax[2].set_ylabel('psi_z')

ax[3].set_ylabel('psi_z scale')


ax[4].set_ylabel('ELBO')

ax[5].set_ylabel('Predictive LL')

ax[5].set_xlabel('Time')

ax[0].legend(loc='upper right')



plt.savefig(f'movielens_test_data_{K}.png')