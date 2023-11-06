import pickle

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

from alan.experiment_utils import seed_torch, n_mean

ml_colour = '#238b45'
linestyles = [':','-']
vi_colour = '##d94801'
rws_colour = '#2171b5'
vi_single_colour = '#6a51a3'
decay_colour = '#000000'

time_mean = 25
Ks = [3,5,10]
num_iters_lst = [2000, 2000, 4000]
uppers_elbo = [-800, -5500, -72500]
lowers_elbo = [-1800, -7000, -77500]

uppers_pred_ll = [-800, -5500, -390000]
lowers_pred_ll = [-1800, -7000, -405000]

exp_lr = None
lrs = [0.01,0.05,0.1]

for smooth in [True, False]:
    if smooth:
        time_mean = 25
    else:
        time_mean = 1
    for adjust_scale in [True, False]:
        for model in range(3):
            num_iters = num_iters_lst[model]
            upp = uppers_elbo[model]
            low = lowers_elbo[model]
            upp_pred = uppers_pred_ll[model]
            low_pred = lowers_pred_ll[model]
            
            model = ['bus_breakdown', 'movielens', 'occupancy'][model]

            for K in Ks:
                fig_iters, ax_iters = plt.subplots(1,2, figsize=(16.0, 7.0))
                # fig_scales, ax_scales = plt.subplots(3,8, figsize=(14.0, 9.0))

                ml1_elbos = None
                ml1_scales = None
                ml1_weights = None

                ### ML
                try:
                    for lr in range(len(lrs)):
                        for ml in [2]:
                            file = f'{model}/results/{model}/ML_{ml}_iters_{num_iters}_lr_{lrs[lr]}_{adjust_scale}_K{K}.pkl'

                            with open(file, 'rb') as f:
                                results_dict = pickle.load(f)

                            elbos = results_dict['objs'].mean(0)
                            pred_lls = results_dict['pred_likelihood'].mean(0)
                            times = results_dict['times'].mean(0)

                            scales = results_dict['scales']
                            weights = results_dict['weights']

                            non_zero_weights = results_dict['non_zero_weights'].mean(0)


                            elbos = np.expand_dims(np.array(elbos), axis=0)
                            pred_lls = np.expand_dims(np.array(pred_lls), axis=0)

                            non_zero_weights = np.expand_dims(np.array(non_zero_weights), axis=0)

                            elbos = n_mean(elbos,time_mean).squeeze(0)
                            pred_lls = n_mean(pred_lls,time_mean).squeeze(0)
                            non_zero_weights = n_mean(non_zero_weights,time_mean).squeeze(0)
                            elbo_lim = elbos.shape[0]

                            ax_iters[0].plot(np.arange(0,num_iters, time_mean), elbos, color=ml_colour, alpha=(1/(3-lr)), label=f'ML {ml} lr: {lrs[lr]}', linestyle=linestyles[ml-1])
                            ax_iters[1].plot(np.arange(0,num_iters, time_mean), pred_lls, color=ml_colour, alpha=(1/(3-lr)), linestyle=linestyles[ml-1])
                except:
                    None

                ### ML decay
                try:
                    for ml in [2]:
                        file = f'{model}/results/{model}/ML_{ml}_iters_{num_iters}_lr_{True}_{adjust_scale}_K{K}.pkl'

                        with open(file, 'rb') as f:
                            results_dict = pickle.load(f)

                        elbos = results_dict['objs'].mean(0)
                        pred_lls = results_dict['pred_likelihood'].mean(0)
                        times = results_dict['times'].mean(0)

                        scales = results_dict['scales']
                        weights = results_dict['weights']

                        non_zero_weights = results_dict['non_zero_weights'].mean(0)


                        elbos = np.expand_dims(np.array(elbos), axis=0)
                        pred_lls = np.expand_dims(np.array(pred_lls), axis=0)

                        non_zero_weights = np.expand_dims(np.array(non_zero_weights), axis=0)

                        elbos = n_mean(elbos,time_mean).squeeze(0)
                        pred_lls = n_mean(pred_lls,time_mean).squeeze(0)
                        non_zero_weights = n_mean(non_zero_weights,time_mean).squeeze(0)
                        elbo_lim = elbos.shape[0]

                        ax_iters[0].plot(np.arange(0,num_iters, time_mean), elbos, color=decay_colour, label=f'ML {ml} Decay', linestyle=linestyles[ml-1])
                        ax_iters[1].plot(np.arange(0,num_iters, time_mean), pred_lls, color=decay_colour, linestyle=linestyles[ml-1])
                except:
                    None

            

        


                try:
                    #### VI
                    for lr in range(len(lrs)):
                        file = f'{model}/results/{model}/VI_{num_iters}_{lrs[lr]}_{adjust_scale}_K{K}.pkl'

                        with open(file, 'rb') as f:
                            results_dict = pickle.load(f)

                        elbos = results_dict['objs'].mean(0)
                        pred_lls = results_dict['pred_likelihood'].mean(0)
                        times = results_dict['times'].mean(0)

                        scales = results_dict['scales']
                        weights = results_dict['weights']

                        non_zero_weights = results_dict['non_zero_weights'].mean(0)


                        elbos = np.expand_dims(np.array(elbos), axis=0)
                        pred_lls = np.expand_dims(np.array(pred_lls), axis=0)
                        non_zero_weights = np.expand_dims(np.array(non_zero_weights), axis=0)


                        ax_iters[0].plot(np.arange(0,num_iters, time_mean), n_mean(elbos,time_mean).squeeze(0), color=vi_colour, alpha=(1/(3-lr)), label=f'VI lr: {lrs[lr]}')
                        ax_iters[1].plot(np.arange(0,num_iters, time_mean), n_mean(pred_lls,time_mean).squeeze(0), color=vi_colour, alpha=(1/(3-lr)))
                except:
                    None

                try:         
                    #### VI single
                    for lr in range(len(lrs)):
                        file = f'{model}/results/{model}/VI_{num_iters}_{lrs[lr]}_{adjust_scale}_K{1}.pkl'

                        with open(file, 'rb') as f:
                            results_dict = pickle.load(f)

                        elbos = results_dict['objs'].mean(0)
                        pred_lls = results_dict['pred_likelihood'].mean(0)
                        times = results_dict['times'].mean(0)

                        scales = results_dict['scales']
                        weights = results_dict['weights']

                        non_zero_weights = results_dict['non_zero_weights'].mean(0)


                        elbos = np.expand_dims(np.array(elbos), axis=0)
                        pred_lls = np.expand_dims(np.array(pred_lls), axis=0)
                        non_zero_weights = np.expand_dims(np.array(non_zero_weights), axis=0)


                        ax_iters[0].plot(np.arange(0,num_iters, time_mean), n_mean(elbos,time_mean).squeeze(0), color=vi_single_colour, alpha=(1/(3-lr)), label=f'VI single sample lr: {lrs[lr]}')
                        ax_iters[1].plot(np.arange(0,num_iters, time_mean), n_mean(pred_lls,time_mean).squeeze(0), color=vi_single_colour, alpha=(1/(3-lr)))
                        # ax_iters[2].plot(np.arange(0,num_iters, time_mean), n_mean(non_zero_weights,time_mean).squeeze(0), color=vi_colours[lr])
                except:
                    None

                try:
                        #### RWS
                    for lr in range(len(lrs)):
                        file = f'{model}/results/{model}/RWS_{num_iters}_{lrs[lr]}_{adjust_scale}_K{K}.pkl'

                        with open(file, 'rb') as f:
                            results_dict = pickle.load(f)

                        elbos = results_dict['objs'].mean(0)
                        pred_lls = results_dict['pred_likelihood'].mean(0)
                        times = results_dict['times'].mean(0)

                        scales = results_dict['scales']
                        weights = results_dict['weights']

                        non_zero_weights = results_dict['non_zero_weights'].mean(0)


                        elbos = np.expand_dims(np.array(elbos), axis=0)
                        pred_lls = np.expand_dims(np.array(pred_lls), axis=0)
                        non_zero_weights = np.expand_dims(np.array(non_zero_weights), axis=0)


                        ax_iters[0].plot(np.arange(0,num_iters, time_mean), n_mean(elbos,time_mean).squeeze(0), color=rws_colour, alpha=(1/(3-lr)), label=f'RWS lr: {lrs[lr]}')
                        ax_iters[1].plot(np.arange(0,num_iters, time_mean), n_mean(pred_lls,time_mean).squeeze(0), color=rws_colour, alpha=(1/(3-lr)))
                        # ax_iters[2].plot(np.arange(0,num_iters, time_mean), n_mean(non_zero_weights,time_mean).squeeze(0), color=vi_colours[lr])


                        
                except:
                    None  



                ax_iters[1].set_title(f'K: {K}, Smoothed over {time_mean} iters, Adjust scale: {adjust_scale}')
                ax_iters[0].set_ylabel('ELBO')
                ax_iters[0].set_ylim(low,upp)
                ax_iters[1].set_ylim(low_pred,upp_pred)
                ax_iters[1].set_ylabel('Predictive LL')
                for i in range(2):
                    ax_iters[i].set_xlabel('Iters')
                fig_iters.legend(loc='upper left')


                fig_iters.tight_layout()
                fig_iters.savefig(f'figures/{model}/{K}_adjust_scale_{adjust_scale}_smoothed_{smooth}.png')

                fig_iters.clf()