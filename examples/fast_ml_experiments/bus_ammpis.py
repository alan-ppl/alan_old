import torch as t
import torch.nn as nn
import pickle
import alan
import alan.postproc as pp


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import time


from alan.experiment_utils import seed_torch, n_mean

from bus_breakdown.bus_breakdown_ammpis import generate_model as generate_AMMP_IS
from bus_breakdown.bus_breakdown_VI import generate_model as generate_VI

ml_colours = ['#ffffb2','#fecc5c','#fd8d3c','#f03b20'][::-1]
vi_colours = ['#edf8fb','#b2e2e2','#66c2a4','#2ca10f'][::-1]

seed_torch(0)
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

M = 3
J = 3
I = 30  
N=0

for use_data in [True]:

    P, Q_ammpis, data, covariates, all_data, all_covariates, sizes = generate_AMMP_IS(N,M, 0, use_data)
    P, Q_vi, _, _, _, _, _ = generate_VI(N,M, device, 2, 0, use_data)


    # # True var
    # z_scale_mean = data['mu_beta']

    # # True mean
    # z_mean = data['sigma_beta']

    with open(f'posteriors/mu_beta_posterior_mean_{use_data}.pkl', 'rb') as f:
        z_scale_mean = pickle.load(f).item()

    # True posterior mean
    with open(f'posteriors/sigma_beta_mean_{use_data}.pkl', 'rb') as f:
        z_mean = pickle.load(f).item()

    data = {'obs':data.pop('obs')}

    K = 10
    T = 2000
    ml_lrs = [0.6]
    vi_lrs = [0.1]

    fig, ax = plt.subplots(4,1, figsize=(7, 8.0))
    for j in range(len(ml_lrs)):
        lr = ml_lrs[j]
        print(f'AMMP-IS: {lr}')
        z_means = []
        z_scales = []
        z_scale_means = []
        z_scale_scales = []
        elbos = []
        times = []
        pred_lls = []
        seed_torch(0)
        q = Q_ammpis()
        m1 = alan.Model(P, q).condition(data=data)

        # samp = m1.sample_same(K, reparam=False)
        # mean_post = mean(samp.weights(1000))
        # var_post

        for i in range(T):
            sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)
            z_means.append(pp.mean(sample.weights()['sigma_beta']).item())    

            z_scale_means.append(pp.mean(sample.weights()['mu_beta']).item())  
            elbos.append(sample.elbo().item()) 
            if i % 100 == 0:
                # print(q.Nz.mean2conv(*q.Nz.named_means))
                print(f'Elbo: {elbos[-1]}')   
            
            start = time.time()    
            m1.ammpis_update(lr, sample)
            times.append(time.time() - start)


            sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)
            pred_lls.append(m1.predictive_ll(sample, N = 10, inputs_all=all_covariates, data_all=all_data)['obs'])


        elbos = np.expand_dims(np.array(elbos), axis=0)
        pred_lls = np.expand_dims(np.array(pred_lls), axis=0)

        z_means = np.expand_dims(np.array(z_means), axis=0)
        z_scale_means = np.expand_dims(np.array(z_scale_means), axis=0)

        ax[0].plot(np.cumsum(times)[::25], n_mean(z_means,25).squeeze(0), color=ml_colours[j], label=f'AMMP-IS lr: {lr}')
        ax[0].axhline(z_mean)
        ax[1].axhline(z_scale_mean)
        ax[1].plot(np.cumsum(times)[::25], n_mean(z_scale_means,25).squeeze(0), color=ml_colours[j])
        ax[2].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=ml_colours[j])
        ax[3].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=ml_colours[j])
        # ax[2].plot(np.cumsum(times), elbos.squeeze(0), color=ml_colours[j])
        # ax[3].plot(np.cumsum(times), pred_lls.squeeze(0), color=ml_colours[j])

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
            z_means.append(pp.mean(sample.weights()['sigma_beta']).item())    

            z_scale_means.append(pp.mean(sample.weights()['mu_beta']).item())  
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

        z_means = np.expand_dims(np.array(z_means), axis=0)
        z_scale_means = np.expand_dims(np.array(z_scale_means), axis=0)

        ax[0].plot(np.cumsum(times)[::25], n_mean(z_means,25).squeeze(0), color=vi_colours[j], label=f'VI lr: {lr}')
        ax[0].axhline(z_mean)
        ax[1].axhline(z_scale_mean)
        ax[1].plot(np.cumsum(times)[::25], n_mean(z_scale_means,25).squeeze(0), color=vi_colours[j])
        ax[2].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=vi_colours[j])
        ax[3].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=vi_colours[j])
        # ax[2].plot(np.cumsum(times), elbos.squeeze(0), color=vi_colours[j])
        # ax[3].plot(np.cumsum(times), pred_lls.squeeze(0), color=vi_colours[j])


    ax[0].set_title(f'K: {K}, Smoothed, Use data: {use_data}')

    ax[0].set_ylabel('mu_beta')


    ax[1].set_ylabel('sigma_beta')




    ax[2].set_ylabel('ELBO')
    # ax[2].set_ylim(-3500,-2900)
    ax[3].set_ylabel('Predictive LL')
    # ax[3].set_ylim(-4000,-3100)
    ax[3].set_xlabel('Time')

    ax[0].legend(loc='upper right')



    plt.savefig(f'figures/bus_ammpis_{K}_{use_data}.png')