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

from movielens.movielens import generate_model as generate_ML
from movielens.movielens_VI import generate_model as generate_VI

decay_colours = ['#9e9ac8','#756bb1','#54278f']
ml_colours = ['#de2d26']
vi_colours = ['#31a354']

seed_torch(0)
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

M=300
N=5

for ML in [2,1]:
    for use_data in [True]:

        P, Q_ml, data, covariates, all_data, all_covariates, sizes = generate_ML(N,M, device, ML, 0, use_data)
        P, Q_vi, _, _, _, _, _ = generate_VI(N,M, device, 2, 0, use_data)


        # # True var
        # z_scale_mean = data['mu_beta']

        # # True mean
        # z_mean = data['sigma_beta']

        # with open(f'posteriors/mu_beta_posterior_mean_{use_data}.pkl', 'rb') as f:
        #     z_scale_mean = pickle.load(f).item()

        # # True posterior mean
        # with open(f'posteriors/sigma_beta_mean_{use_data}.pkl', 'rb') as f:
        #     z_mean = pickle.load(f).item()

        data = {'obs':data.pop('obs')}

        for K in [5,10]:
            T = 7500
            ml_lrs = [1, 0.9,0.75]
            vi_lrs = [0.1]

            # fig_times, ax_times = plt.subplots(1,3, figsize=(12.0, 7.0))
            fig_iters, ax_iters = plt.subplots(1,3, figsize=(12.0, 7.0))
            fig_scales, ax_scales = plt.subplots(2,3, figsize=(12.0, 6.0))

            #ML with decay LR
            print('ML Decay')
            for j in range(len(ml_lrs)):
                exp_lr = ml_lrs[j]


                elbos = []
                times = []
                pred_lls = []

                non_zero_weights = []
                seed_torch(0)
                q = Q_ml()
                m1 = alan.Model(P, q).condition(data=data)

                samp = m1.sample_same(K, inputs=covariates, reparam=False, device=device)

                scales = {k:[] for k in samp.weights().keys()}
                weights = {k:[] for k in samp.weights().keys()}


                try:
                    for i in range(T):
                        lr = ((i + 10)**(-exp_lr))
                        
                        sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)
                        elbos.append(sample.elbo().item()) 

                        non_zero_weight = 0  
                        num_latents = 0
                            
                        for k,v in sample.weights().items():
                            scales[k].append(q.__getattr__(k).mean2conv(*q.__getattr__(k).named_means)['scale'].mean())
                            weights[k].append((v[1].rename(None) > 0.001).sum())

                            non_zero_weight += (v[1].rename(None) > 0.001).sum()
                            num_latents += v[0].numel()

                        non_zero_weights.append(non_zero_weight)

                        if i % 100 == 0:
                            # print(q.Nz.mean2conv(*q.Nz.named_means))

                            print(f'Iteration: {i}, Elbo: {elbos[-1]}, lr: {lr}') 
                        
                        start = time.time()    
                        m1.update(lr, sample)
                        times.append(time.time() - start)


                        sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)
                        try:
                            pred_lls.append(m1.predictive_ll(sample, N = 10, inputs_all=all_covariates, data_all=all_data)['obs'])
                        except:
                            pred_lls.append(np.nan)
                except:
                    None


                elbos = np.expand_dims(np.array(elbos), axis=0)
                pred_lls = np.expand_dims(np.array(pred_lls), axis=0)
                
                non_zero_weights = np.expand_dims(np.array(non_zero_weights), axis=0)


                elbos_length = elbos.shape[1]
                pred_lls_length = pred_lls.shape[1]
                non_zero_weights_length = non_zero_weights.shape[1]

                elbos = n_mean(elbos,25).squeeze(0)
                pred_lls = n_mean(pred_lls,25).squeeze(0)
                non_zero_weights = n_mean(non_zero_weights,25).squeeze(0)
                lim = np.cumsum(times)[::1].shape[0]

                # ax_times[0].plot(np.cumsum(times)[::1][:elbos.shape[0]], elbos, color=decay_colours[j], label=f'ML decay: {exp_lr}')
                # ax_times[1].plot(np.cumsum(times)[::1][:pred_lls.shape[0]], pred_lls, color=decay_colours[j])
                # ax_times[2].plot(np.cumsum(times)[::1][:non_zero_weights.shape[0]], non_zero_weights, color=decay_colours[j])

                ax_iters[0].plot(np.arange(elbos_length)[::25], elbos, color=decay_colours[j], label=f'ML decay: {exp_lr}')
                ax_iters[1].plot(np.arange(pred_lls_length)[::25], pred_lls, color=decay_colours[j])
                ax_iters[2].plot(np.arange(non_zero_weights_length)[::25], non_zero_weights, color=decay_colours[j])


                for i, (k,v) in enumerate(scales.items()):
                    v = np.expand_dims(np.array(v), axis=0)
                    if i == 0:
                        ax_scales[0,i].plot(v.squeeze(0), color=decay_colours[j], label=f'ML decay: {exp_lr}')
                    else:
                        ax_scales[0,i].plot(v.squeeze(0), color=decay_colours[j])
                        
                    ax_scales[0,i].set_title(k)


                for i, (k,v) in enumerate(weights.items()):
                    v = np.expand_dims(np.array(v), axis=0)
                    if i == 0:
                        ax_scales[1,i].plot(v.squeeze(0), color=decay_colours[j])
                    else:
                        ax_scales[1,i].plot(v.squeeze(0), color=decay_colours[j])
                        
                    ax_scales[1,i].set_title(k)

            #ML
            print(f'ML: 0.001')

            elbos = []
            times = []
            pred_lls = []

            non_zero_weights = []

            seed_torch(0)
            q = Q_ml()
            m1 = alan.Model(P, q).condition(data=data)

            samp = m1.sample_same(K, inputs=covariates, reparam=False, device=device)

            scales = {k:[] for k in samp.weights().keys()}
            weights = {k:[] for k in samp.weights().keys()}
            try:
                for i in range(T):
                    lr = 0.001
                    
                    sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)

                    elbos.append(sample.elbo().item()) 

                    non_zero_weight = 0  
                    num_latents = 0
                        
                    for k,v in sample.weights().items():
                        scales[k].append(q.__getattr__(k).mean2conv(*q.__getattr__(k).named_means)['scale'].mean())
                        weights[k].append((v[1].rename(None) > 0.001).sum())

                        non_zero_weight += (v[1].rename(None) > 0.001).sum()
                        num_latents += v[0].numel()

                    non_zero_weights.append(non_zero_weight)

                    if i % 100 == 0:
                        # print(q.Nz.mean2conv(*q.Nz.named_means))

                        print(f'Iteration: {i}, Elbo: {elbos[-1]}, lr: {lr}')   
                    
                    start = time.time()    
                    m1.update(lr, sample)
                    times.append(time.time() - start)


                    sample = m1.sample_same(K, inputs=covariates, reparam=False, device=device)
                    try:
                        pred_lls.append(m1.predictive_ll(sample, N = 10, inputs_all=all_covariates, data_all=all_data)['obs'])
                    except:
                        pred_lls.append(np.nan)
            except:
                None

            elbos = np.expand_dims(np.array(elbos), axis=0)
            pred_lls = np.expand_dims(np.array(pred_lls), axis=0)
            non_zero_weights = np.expand_dims(np.array(non_zero_weights), axis=0)




            elbos_length = elbos.shape[1]
            pred_lls_length = pred_lls.shape[1]
            non_zero_weights_length = non_zero_weights.shape[1]
            
            elbos = n_mean(elbos,25).squeeze(0)
            pred_lls = n_mean(pred_lls,25).squeeze(0)
            non_zero_weights = n_mean(non_zero_weights,25).squeeze(0)
            lim = np.cumsum(i)[::1].shape[0]

            # ax_times[0].plot(np.cumsum(times)[::1], elbos, color=ml_colours[0], label=f'ML lr: {lr}')
            # ax_times[1].plot(np.cumsum(times)[::1], pred_lls, color=ml_colours[0])
            # ax_times[2].plot(np.cumsum(times)[::1], non_zero_weights, color=ml_colours[0])

            ax_iters[0].plot(np.arange(elbos_length)[::25], elbos, color=ml_colours[0], label=f'ML lr: {lr}')
            ax_iters[1].plot(np.arange(pred_lls_length)[::25], pred_lls, color=ml_colours[0])
            ax_iters[2].plot(np.arange(non_zero_weights_length)[::25], non_zero_weights, color=ml_colours[0])

            for i, (k,v) in enumerate(scales.items()):
                v = np.expand_dims(np.array(v), axis=0)
                if i == 0:
                    ax_scales[0,i].plot(v.squeeze(0), color=ml_colours[0], label=f'ML lr: {lr}')
                else:
                    ax_scales[0,i].plot(v.squeeze(0), color=ml_colours[0])

                ax_scales[0,i].set_title(k)

            for i, (k,v) in enumerate(weights.items()):
                v = np.expand_dims(np.array(v), axis=0)
                if i == 0:
                    ax_scales[1,i].plot(v.squeeze(0), color=ml_colours[0])
                else:
                    ax_scales[1,i].plot(v.squeeze(0), color=ml_colours[0])
                    
                ax_scales[1,i].set_title(k)
            #VI   
            print('VI')
            for j in range(len(vi_lrs)):
                lr = vi_lrs[j]
                print(f'VI: {lr}')


                elbos = []
                times = []
                pred_lls = []
                non_zero_weights = []
                seed_torch(0)
                q = Q_vi()
                cond_model = alan.Model(P, q).condition(data=data)
                opt = t.optim.Adam(cond_model.parameters(), lr=lr)

                samp = cond_model.sample_same(K, inputs=covariates, reparam=False, device=device)

                scales = {k:[] for k in samp.weights().keys()}
                weights = {k:[] for k in samp.weights().keys()}

                for i in range(T):
                    opt.zero_grad()
                    sample = cond_model.sample_same(K, reparam=True, inputs=covariates, device=device)
    
                    elbo = sample.elbo()
                    elbos.append(elbo.item())
                    
                    start = time.time()
                    (-elbo).backward()
                    opt.step()
                    times.append(time.time() - start)


                    sample = cond_model.sample_same(K, reparam=False, inputs=covariates, device=device)
                    pred_lls.append(cond_model.predictive_ll(sample, N = 10, inputs_all=all_covariates, data_all=all_data)['obs'])



                    non_zero_weight = 0  
                    num_latents = 0
                        
                    for k,v in sample.weights().items():
                        weights[k].append((v[1].rename(None) > 0.001).sum())

                        non_zero_weight += (v[1].rename(None) > 0.001).sum()
                        num_latents += v[0].numel()

                        k_sigma = 'log_' + k + '_sigma'
                        sc = q.__getattr__(k_sigma).clone().detach()
                        if hasattr(sc, "dims"):
                            sc = sc.order(*sc.dims)
                        scales[k].append(sc.exp().mean().item())
                    
                    non_zero_weights.append(non_zero_weight)
                        
                    if i % 100 == 0:
                        print(f'Iteration: {i}, Elbo: {elbos[-1]}, lr: {lr}')        


                elbos = np.expand_dims(np.array(elbos), axis=0)
                pred_lls = np.expand_dims(np.array(pred_lls), axis=0)
                non_zero_weights = np.expand_dims(np.array(non_zero_weights), axis=0)


                elbos_length = elbos.shape[1]
                pred_lls_length = pred_lls.shape[1]
                non_zero_weights_length = non_zero_weights.shape[1]


                elbos = n_mean(elbos,25).squeeze(0)
                pred_lls = n_mean(pred_lls,25).squeeze(0)
                non_zero_weights = n_mean(non_zero_weights,25).squeeze(0)
                lim = np.cumsum(i)[::1].shape[0]

                # ax_times[0].plot(np.cumsum(times)[::1], n_mean(elbos,1).squeeze(0), color=vi_colours[j], label=f'VI lr: {lr}')
                # ax_times[1].plot(np.cumsum(times)[::1], n_mean(pred_lls,1).squeeze(0), color=vi_colours[j])
                # ax_times[2].plot(np.cumsum(times)[::1], n_mean(non_zero_weights,1).squeeze(0), color=vi_colours[j])

                ax_iters[0].plot(np.arange(elbos_length)[::25], elbos, color=vi_colours[j], label=f'VI lr: {lr}')
                ax_iters[1].plot(np.arange(pred_lls_length)[::25], pred_lls, color=vi_colours[j])
                ax_iters[2].plot(np.arange(non_zero_weights_length)[::25], non_zero_weights, color=vi_colours[j])

                for i, (k,v) in enumerate(scales.items()):
                    v = np.expand_dims(np.array(v), axis=0)
                    if i == 0:
                        ax_scales[0,i].plot(v.squeeze(0), color=vi_colours[j], label=f'VI lr: {lr}')
                    else:
                        ax_scales[0,i].plot(v.squeeze(0), color=vi_colours[j])
                    ax_scales[0,i].set_title(k)


                for i, (k,v) in enumerate(weights.items()):
                    v = np.expand_dims(np.array(v), axis=0)
                    if i == 0:
                        ax_scales[1,i].plot(v.squeeze(0), color=vi_colours[j])
                    else:
                        ax_scales[1,i].plot(v.squeeze(0), color=vi_colours[j])
                        
                    ax_scales[1,i].set_title(k)




            # ax_times[1].set_title(f'K: {K}, Smoothed, Use data: {use_data}')
            # ax_times[0].set_ylabel('ELBO')
            # ax_times[0].set_ylim(-4500,-1000)
            # ax_times[1].set_ylabel('Predictive LL')
            # ax_times[2].set_ylabel('Non zero weights')
            # ax_times[1].set_xlabel('Time')
            # fig_times.legend(loc='upper right')


            # fig_times.tight_layout()
            # fig_times.savefig(f'figures/movielens_test_data_{K}_{use_data}_elbo_ML_{ML}.png')


            ax_iters[1].set_title(f'K: {K}, Smoothed, Use data: {use_data}')
            ax_iters[0].set_ylabel('ELBO')
            ax_iters[0].set_ylim(-4000,-1000)
            ax_iters[1].set_ylabel('Predictive LL')
            ax_iters[2].set_ylabel('Non zero weights')
            ax_iters[1].set_xlabel('Iters')
            fig_iters.legend(loc='upper right')


            fig_iters.tight_layout()
            fig_iters.savefig(f'figures/movielens_test_data_{K}_{use_data}_elbo_iters_ML_{ML}.png')

            ax_scales[0,0].set_ylabel('Mean Scale')
            ax_scales[1,0].set_ylabel('Mean Non zero weights')
            fig_scales.legend(loc='upper right')
            fig_scales.tight_layout()
            fig_scales.savefig(f'figures/bus_test_data_{K}_{use_data}_scales_weights_ML_{ML}.png')

