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

from bus_breakdown.bus_breakdown import generate_model as generate_ML
from bus_breakdown.bus_breakdown_VI import generate_model as generate_VI

adaptive_colours = ['#bcbddc','#756bb1']
ml_colours = ['#de2d26']
vi_colours = ['#31a354']

seed_torch(0)
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

M = 3
J = 3
I = 30  
N=0

for use_data in [True]:

    P, Q_ml, data, covariates, all_data, all_covariates, sizes = generate_ML(N,M, device, 2, 0, use_data)
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

    for K in [5,10]:
        T = 1000
        ml_lrs = [0.9,0.75]
        vi_lrs = [0.1]

        fig, ax = plt.subplots(4,1, figsize=(7, 8.0))
        fig2, ax2 = plt.subplots(1,3, figsize=(7.0, 5.0))

        # #ML with adaptive LR
        # print('ML adaptive')
        # for j in range(len(ml_lrs)):
        #     exp_lr = ml_lrs[j]

        #     z_means = []
        #     z_scales = []
        #     z_scale_means = []
        #     z_scale_scales = []

        #     scales = []
        #     elbos = []
        #     times = []
        #     pred_lls = []
        #     seed_torch(0)
        #     q = Q_ml()
        #     m1 = alan.Model(P, q).condition(data=data)

        #     # samp = m1.sample_same(K, reparam=False)
        #     # mean_post = mean(samp.weights(1000))
        #     # var_post
        #     try:
        #         for i in range(T):
        #             lr = ((i + 10)**(-exp_lr))/2
                    
        #             sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)
        #             z_means.append(pp.mean(sample.weights()['sigma_beta']).item())    

        #             z_scale_means.append(pp.mean(sample.weights()['mu_beta']).item())  
        #             elbos.append(sample.elbo().item()) 
        #             scale = 0
                    
        #             scale += q.sigma_beta.mean2conv(*q.sigma_beta.named_means)['scale'].sum()
        #             # scale += q.mu_beta.mean2conv(*q.mu_beta.named_means)['scale'].sum()
        #             # scale += q.beta.mean2conv(*q.beta.named_means)['scale'].sum()
        #             # scale += q.alpha.mean2conv(*q.alpha.named_means)['scale'].sum()
        #             # scale += q.sigma_alpha.mean2conv(*q.sigma_alpha.named_means)['scale'].sum()
        #             # scale += q.psi.mean2conv(*q.psi.named_means)['scale'].sum()
        #             # scale += q.phi.mean2conv(*q.phi.named_means)['scale'].sum()
        #             # scale += q.log_sigma_phi_psi.mean2conv(*q.log_sigma_phi_psi.named_means)['scale'].sum()

        #             scales.append(scale)
        #             if i % 100 == 0:
        #                 # print(q.Nz.mean2conv(*q.Nz.named_means))

        #                 print(f'Elbo: {elbos[-1]}, lr: {lr}')   
                    
        #             start = time.time()    
        #             m1.update(lr, sample)
        #             times.append(time.time() - start)


        #             sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)
        #             try:
        #                 pred_lls.append(m1.predictive_ll(sample, N = 10, inputs_all=all_covariates, data_all=all_data)['obs'])
        #             except:
        #                 pred_lls.append(np.nan)
        #     except:
        #         z_means.append(np.nan)    

        #         z_scale_means.append(np.nan)  
        #         elbos.append(np.nan) 
        #         pred_lls.append(np.nan)
        #         scales.append(np.nan)
        #         times.append(np.nan)


        #     elbos = np.expand_dims(np.array(elbos), axis=0)
        #     pred_lls = np.expand_dims(np.array(pred_lls), axis=0)
        #     scales = np.expand_dims(np.array(scales), axis=0)

        #     z_means = np.expand_dims(np.array(z_means), axis=0)
        #     z_scale_means = np.expand_dims(np.array(z_scale_means), axis=0)

        #     lim = np.cumsum(times)[::25].shape[0]
        #     ax[0].plot(np.cumsum(times)[::25], n_mean(z_means,25).squeeze(0), color=adaptive_colours[j], label=f'ML adaptive: {exp_lr}')
        #     ax[0].axhline(z_mean)
        #     ax[1].axhline(z_scale_mean)
        #     ax[1].plot(np.cumsum(times)[::25], n_mean(z_scale_means,25).squeeze(0), color=adaptive_colours[j])
        #     ax[2].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=adaptive_colours[j])
        #     ax2[0].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=adaptive_colours[j], label=f'ML adaptive: {exp_lr}')
        #     ax[3].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=adaptive_colours[j])
        #     ax2[1].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=adaptive_colours[j])
        #     ax2[2].plot(np.cumsum(times), scales.squeeze(0), color=adaptive_colours[j])
        #     # ax[2].plot(np.cumsum(times), elbos.squeeze(0), color=ml_colours[j])
        #     # ax[3].plot(np.cumsum(times), pred_lls.squeeze(0), color=ml_colours[j])

        #ML
        print(f'ML: 0.001')

        z_means = []
        z_scales = []
        z_scale_means = []
        z_scale_scales = []

        scales = []
        elbos = []
        times = []
        pred_lls = []
        seed_torch(0)
        q = Q_ml()
        m1 = alan.Model(P, q).condition(data=data)

        # samp = m1.sample_same(K, reparam=False)
        # mean_post = mean(samp.weights(1000))
        # var_post
        try:
            for i in range(T):
                lr = 0.001
                
                sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)
                z_means.append(pp.mean(sample.weights()['sigma_beta']).item())    

                z_scale_means.append(pp.mean(sample.weights()['mu_beta']).item())  
                elbos.append(sample.elbo().item()) 
                scale = 0
                
                scale += q.sigma_beta.mean2conv(*q.sigma_beta.named_means)['scale'].sum()
                # scale += q.mu_beta.mean2conv(*q.mu_beta.named_means)['scale'].sum()
                # scale += q.beta.mean2conv(*q.beta.named_means)['scale'].sum()
                # scale += q.alpha.mean2conv(*q.alpha.named_means)['scale'].sum()
                # scale += q.sigma_alpha.mean2conv(*q.sigma_alpha.named_means)['scale'].sum()
                # scale += q.psi.mean2conv(*q.psi.named_means)['scale'].sum()
                # scale += q.phi.mean2conv(*q.phi.named_means)['scale'].sum()
                # scale += q.log_sigma_phi_psi.mean2conv(*q.log_sigma_phi_psi.named_means)['scale'].sum()

                scales.append(scale)
                if i % 100 == 0:
                    # print(q.Nz.mean2conv(*q.Nz.named_means))

                    print(f'Elbo: {elbos[-1]}, lr: {lr}')   
                
                start = time.time()    
                m1.update(lr, sample)
                times.append(time.time() - start)


                sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)
                try:
                    pred_lls.append(m1.predictive_ll(sample, N = 10, inputs_all=all_covariates, data_all=all_data)['obs'])
                except:
                    pred_lls.append(np.nan)
        except:
            z_means.append(np.nan)    

            z_scale_means.append(np.nan)  
            elbos.append(np.nan) 
            pred_lls.append(np.nan)
            scales.append(np.nan)
            times.append(np.nan)

        elbos = np.expand_dims(np.array(elbos), axis=0)
        pred_lls = np.expand_dims(np.array(pred_lls), axis=0)
        scales = np.expand_dims(np.array(scales), axis=0)

        z_means = np.expand_dims(np.array(z_means), axis=0)
        z_scale_means = np.expand_dims(np.array(z_scale_means), axis=0)

        lim = np.cumsum(times)[::25].shape[0]
        ax[0].plot(np.cumsum(times)[::25], n_mean(z_means,25).squeeze(0), color=ml_colours[0], label=f'ML lr: {lr}')
        ax[0].axhline(z_mean)
        ax[1].axhline(z_scale_mean)
        ax[1].plot(np.cumsum(times)[::25], n_mean(z_scale_means,25).squeeze(0), color=ml_colours[0])
        ax[2].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=ml_colours[0])
        ax[3].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=ml_colours[0])

        ax2[0].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=ml_colours[0], label=f'ML lr: {lr}')
        ax2[1].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=ml_colours[0])
        ax2[2].plot(np.cumsum(times), scales.squeeze(0), color=ml_colours[0])
        # ax[2].plot(np.cumsum(times), elbos.squeeze(0), color=ml_colours[j])
        # ax[3].plot(np.cumsum(times), pred_lls.squeeze(0), color=ml_colours[j])

        #VI   
        print('VI')
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

            ax2[0].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=vi_colours[j], label=f'VI lr: {lr}')
            ax2[1].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=vi_colours[j])
            # ax[2].plot(np.cumsum(times), elbos.squeeze(0), color=vi_colours[j])
            # ax[3].plot(np.cumsum(times), pred_lls.squeeze(0), color=vi_colours[j])


        ax[0].set_title(f'K: {K}, Smoothed, Use data: {use_data}')
        ax2[0].set_title(f'K: {K}, Smoothed, Use data: {use_data}')

        ax[0].set_ylabel('mu_beta')


        ax[1].set_ylabel('sigma_beta')




        ax[2].set_ylabel('ELBO')
        ax2[0].set_ylabel('ELBO')
        # ax[2].set_ylim(-3500,-2900)
        ax[3].set_ylabel('Predictive LL')
        ax2[1].set_ylabel('Predictive LL')

        ax2[2].set_ylabel('Approx posterior Scale')
        # ax[3].set_ylim(-4000,-3100)
        ax[3].set_xlabel('Time')
        ax2[1].set_xlabel('Time')

        ax[0].legend(loc='upper right')
        fig2.legend(loc='upper right')



        fig.savefig(f'figures/bus_test_data_{K}_{use_data}.png')

        fig2.tight_layout()
        fig2.savefig(f'figures/bus_test_data_{K}_{use_data}_elbo.png')