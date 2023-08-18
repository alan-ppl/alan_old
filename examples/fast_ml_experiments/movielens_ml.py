import torch as t
import torch.nn as nn
import pickle 

import alan
import alan.postproc as pp


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#matplotlib.use('TkAgg')
import time


from alan.experiment_utils import seed_torch, n_mean

from movielens.movielens import generate_model as generate_ML
from movielens.movielens_VI import generate_model as generate_VI

adaptive_colours = ['#bcbddc','#756bb1']
ml_colours = ['#fc9272', '#de2d26']
vi_colours = ['#31a354']
seed_torch(0)

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

M=300
N=5

for use_data in [True]:

    P, Q_ml, data, covariates, all_data, all_covariates, sizes = generate_ML(N,M, device, 2, 0, use_data)
    P, Q_vi, _, _, _, _, _ = generate_VI(N,M, device, 2, 0, use_data)

    # # True psi_z
    # z_scale_mean = data['mu_z'][0]

    # # True mean
    # z_mean = data['psi_z'][0]

    # True posterior psi_z

    # with open(f'posteriors/psi_z_posterior_mean_{use_data}.pkl', 'rb') as f:
    #     z_scale_mean = pickle.load(f)


    # # True posterior mean
    # with open(f'posteriors/mu_z_posterior_mean_{use_data}.pkl', 'rb') as f:
    #     z_mean = pickle.load(f)


    data = {'obs':data.pop('obs')}

    for K in [5,10]:
        T =  7500
        adaptive_lrs = [0.9,0.75]
        vi_lrs = [0.1]

        fig, ax = plt.subplots(19,2, figsize=(7.0, 5*9.0))
        fig2, ax2 = plt.subplots(1,3, figsize=(7.0, 5.0))
        for j in range(len(adaptive_lrs)):
            exp_lr = adaptive_lrs[j]
            print('ML adaptive')
            z_means = {i:[] for i in range(18)}

            z_scale_means = {i:[] for i in range(18)}

            elbos = []
            scales = []
            times = []
            pred_lls = []
            seed_torch(0)
            q = Q_ml()
            m1 = alan.Model(P, q).condition(data=data)

            m1.to(device)
            # samp = m1.sample_same(K, reparam=False)
            # mean_post = mean(samp.weights(1000))
            # var_post

            for i in range(T):
                lr = (i + 10)**(-exp_lr)
                sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)

                mns = pp.mean(sample.weights())
                stds = pp.std(sample.weights())
                zm = mns['mu_z']
                zsm = mns['psi_z']

                stdm = stds['mu_z']
                stdsm = stds['psi_z']
                
                scale = q.mu.mean2conv(*q.mu.named_means)['scale'].sum() + q.psi_z.mean2conv(*q.psi_z.named_means)['scale'].sum()
                scale /= q.mu.mean2conv(*q.mu.named_means)['scale'].numel() + q.psi_z.mean2conv(*q.psi_z.named_means)['scale'].numel()

                scales.append(scale.item())
                for k in range(18):

                    z_means[k].append(zm[k].item())    

                    z_scale_means[k].append(zsm[k].item())  

                elbos.append(sample.elbo().item().cpu()) 
                if i % 100 == 0:
                    # print(q.Nz.mean2conv(*q.Nz.named_means))
                    print(f'Elbo: {elbos[-1]}, lr: {lr}')     
                
                start = time.time()    
                m1.update(lr, sample)
                times.append(time.time() - start)


                sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)
                pred_lls.append(m1.predictive_ll(sample, N = 10,inputs_all=all_covariates, data_all=all_data)['obs'].cpu())


            elbos = np.expand_dims(np.array(elbos), axis=0)
            pred_lls = np.expand_dims(np.array(pred_lls), axis=0)

            # for i in range(18):
            #     ax[i,0].plot(np.cumsum(times), z_means[i], color=adaptive_colours[j], label=f'ML adaptive: {exp_lr}')
            #     # ax[i,0].axhline(z_mean[i])
            #     # ax[i,1].axhline(z_scale_mean[i])
            #     ax[i,1].plot(np.cumsum(times), z_scale_means[i], color=adaptive_colours[j])
                # ax[4].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=vi_colours[j])
                # ax[5].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=vi_colours[j])
            # ax[18,0].plot(np.cumsum(times), elbos.squeeze(0), color=adaptive_colours[j])

            # ax[18,1].plot(np.cumsum(times), pred_lls.squeeze(0), color=adaptive_colours[j])

            ax2[0].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=adaptive_colours[j], label=f'ML adaptive: {exp_lr}')
            ax2[1].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=adaptive_colours[j])
            ax2[2].plot(np.cumsum(times), scales, color=adaptive_colours[j])

        ml_lrs = [0.1,0.05]
        for j in range(len(ml_lrs)):
            lr = ml_lrs[j]
            print(f'ML: {lr}')
            z_means = {i:[] for i in range(18)}

            z_scale_means = {i:[] for i in range(18)}

            elbos = []
            scales = []
            times = []
            pred_lls = []
            seed_torch(0)
            q = Q_ml()
            m1 = alan.Model(P, q).condition(data=data)
            m1.to(device)
            # samp = m1.sample_same(K, reparam=False)
            # mean_post = mean(samp.weights(1000))
            # var_post

            for i in range(T):
                sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)

                mns = pp.mean(sample.weights())
                stds = pp.std(sample.weights())
                zm = mns['mu_z']
                zsm = mns['psi_z']

                stdm = stds['mu_z']
                stdsm = stds['psi_z']
                
                scale = q.mu.mean2conv(*q.mu.named_means)['scale'].sum() + q.psi_z.mean2conv(*q.psi_z.named_means)['scale'].sum()
                scale /= q.mu.mean2conv(*q.mu.named_means)['scale'].numel() + q.psi_z.mean2conv(*q.psi_z.named_means)['scale'].numel()

                scales.append(scale.item())
                for k in range(18):

                    z_means[k].append(zm[k].item())    

                    z_scale_means[k].append(zsm[k].item())  

                elbos.append(sample.elbo().item().cpu()) 
                if i % 100 == 0:
                    # print(q.Nz.mean2conv(*q.Nz.named_means))
                    print(f'Elbo: {elbos[-1]}')   
                
                start = time.time()    
                m1.update(lr, sample)
                times.append(time.time() - start)


                sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)
                pred_lls.append(m1.predictive_ll(sample, N = 10,inputs_all=all_covariates, data_all=all_data)['obs'].cpu())


            elbos = np.expand_dims(np.array(elbos), axis=0)
            pred_lls = np.expand_dims(np.array(pred_lls), axis=0)


            ax2[0].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=ml_colours[j], label=f'ML lr: {lr}')
            ax2[1].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=ml_colours[j])
            ax2[2].plot(np.cumsum(times), scales, color=ml_colours[j])

        for j in range(len(vi_lrs)):
            lr = vi_lrs[j]
            print(f'VI: {lr}')
            z_means = {i:[] for i in range(18)}

            z_scale_means = {i:[] for i in range(18)}
            elbos = []
            scales = []
            times = []
            pred_lls = []
            seed_torch(0)
            q = Q_vi()
            cond_model = alan.Model(P, q).condition(data=data)
            cond_model.to(device)
            opt = t.optim.Adam(cond_model.parameters(), lr=lr)
            for i in range(2000):
                opt.zero_grad()
                sample = cond_model.sample_same(K, reparam=True, inputs=covariates, device=device)

                mns = pp.mean(sample.weights())
                zm = mns['mu_z']
                zsm = mns['psi_z']

                for k in range(18):

                    z_means[k].append(zm[k].item())    

                    z_scale_means[k].append(zsm[k].item())  
        
                elbo = sample.elbo()
                elbos.append(elbo.item().cpu())
                
                start = time.time()
                (-elbo).backward()
                opt.step()
                times.append(time.time() - start)


                sample = cond_model.sample_same(K, reparam=False, inputs=covariates, device=device)
                pred_lls.append(cond_model.predictive_ll(sample, N = 10, inputs_all=all_covariates, data_all=all_data)['obs'].cpu())


                if i % 100 == 0:
                    print(f'Elbo: {elbos[-1]}')        


            elbos = np.expand_dims(np.array(elbos), axis=0)
            pred_lls = np.expand_dims(np.array(pred_lls), axis=0)

            # for i in range(18):
            #     ax[i,0].plot(np.cumsum(times), z_means[i], color=vi_colours[j], label=f'VI lr: {lr}')
            #     ax[i,0].axhline(z_mean[i])
            #     ax[i,0].set_ylabel(f'mu_z_{i}')
            #     ax[i,1].axhline(z_scale_mean[i])
            #     ax[i,1].plot(np.cumsum(times), z_scale_means[i], color=vi_colours[j])
            #     ax[i,1].set_ylabel(f'psi_z_{i}')
            #     # ax[4].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=vi_colours[j])
            #     # ax[5].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=vi_colours[j])
            # ax[18,0].plot(np.cumsum(times), elbos.squeeze(0), color=vi_colours[j])
            
            # ax[18,1].plot(np.cumsum(times), pred_lls.squeeze(0), color=vi_colours[j])


            ax2[0].plot(np.cumsum(times)[::25], n_mean(elbos,25).squeeze(0), color=vi_colours[j], label=f'VI lr: {lr}')
            ax2[1].plot(np.cumsum(times)[::25], n_mean(pred_lls,25).squeeze(0), color=vi_colours[j])



        # fig.suptitle(f'K: {K}, Using Data: {use_data}')
        fig2.suptitle(f'K: {K}, Using Data: {use_data}')






        # ax[18,0].set_ylabel('ELBO')
        
        # ax[18,0].set_xlabel('Time')
        # ax[18,1].set_xlabel('Time')

        # ax[18,1].set_ylabel('Predictive LL')

        # ax[0,0].legend(loc='upper right')

        # fig.tight_layout()
        # fig.savefig(f'figures/movielens_test_data_{K}_{use_data}.png')


        ax2[0].set_ylabel('ELBO')
        ax2[1].set_ylabel('Predictive LL')
        
        
        ax2[0].set_xlabel('Time')

        ax2[2].set_ylabel('Approx Posterior scales')

        
        fig2.legend(loc='upper right')



        ax2[0].set_ylim(-1300,-900)
        ax2[1].set_ylim(-1050,-800)



        fig2.tight_layout()
        fig2.savefig(f'figures/movielens_test_data_{K}_{use_data}_elbo.png')