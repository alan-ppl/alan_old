import pickle

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

from alan.experiment_utils import seed_torch, n_mean

decay_colours = ['#9e9ac8','#de2d26']
linestyles = ['-',':']
vi_colours = ['#31a354']

fig_iters, ax_iters = plt.subplots(1,3, figsize=(12.0, 7.0))
fig_scales, ax_scales = plt.subplots(2,8, figsize=(12.0, 6.0))
time_mean = 10
K = 10
num_iters = 5000
exp_lr = 0.9
VI_lr = 0.01
### ML
for ml in [1]:
    file = 'bus_breakdown/results/bus_breakdown/ML_{}'.format(ml)+ '_iters_{}'.format(num_iters) + '_decay_{}_'.format(exp_lr) + 'K{0}.pkl'.format(K)

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
    non_zero_weights = n_mean(non_zero_weights,1).squeeze(0)
    elbo_lim = elbos.shape[0]

    # ax_times[0].plot(times, elbos, color=decay_colours[j], label=f'ML decay: {exp_lr}')
    # ax_times[1].plot(times, pred_lls, color=decay_colours[j])
    # ax_times[2].plot(times, non_zero_weights, color=decay_colours[j])

    ax_iters[0].plot(elbos, color=decay_colours[ml-1], label=f'ML {ml} decay: {exp_lr}', linestyle=linestyles[ml-1])
    ax_iters[1].plot(pred_lls, color=decay_colours[ml-1], linestyle=linestyles[ml-1])
    ax_iters[2].plot(non_zero_weights, color=decay_colours[ml-1], linestyle=linestyles[ml-1])


    for i, (k,v) in enumerate(scales.items()):
        v = v.mean(0)
        v = np.expand_dims(np.array(v), axis=0)
        if i == 0:
            ax_scales[0,i].plot(v.squeeze(0), color=decay_colours[ml-1], label=f'ML {ml} decay: {exp_lr}', linestyle=linestyles[ml-1])
        else:
            ax_scales[0,i].plot(v.squeeze(0), color=decay_colours[ml-1], linestyle=linestyles[ml-1])
        ax_scales[0,i].set_title(k)         



    for i, (k,v) in enumerate(weights.items()):
        v = v.mean(0)
        v = np.expand_dims(np.array(v), axis=0)
        if i == 0:
            ax_scales[1,i].plot(v.squeeze(0), color=decay_colours[ml-1], linestyle=linestyles[ml-1])
        else:
            ax_scales[1,i].plot(v.squeeze(0), color=decay_colours[ml-1], linestyle=linestyles[ml-1])
        ax_scales[1,i].set_title(k)           
        


#### VI
file = 'bus_breakdown/results/bus_breakdown/VI_{}'.format(num_iters) + '_{}_'.format(VI_lr) + 'K{0}.pkl'.format(K)

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


ax_iters[0].plot(n_mean(elbos,time_mean).squeeze(0), color=vi_colours[0], label=f'VI lr: {VI_lr}')
ax_iters[1].plot(n_mean(pred_lls,time_mean).squeeze(0), color=vi_colours[0])
ax_iters[2].plot(n_mean(non_zero_weights,time_mean).squeeze(0), color=vi_colours[0])

for i, (k,v) in enumerate(scales.items()):
    v = v.mean(0)
    v = np.expand_dims(np.array(v), axis=0)
    if i == 0:
        ax_scales[0,i].plot(v.squeeze(0), color=vi_colours[0], label=f'VI lr: {VI_lr}')
    else:
        ax_scales[0,i].plot(v.squeeze(0), color=vi_colours[0])
    ax_scales[0,i].set_title(k)
    


for i, (k,v) in enumerate(weights.items()):
    v = v.mean(0)
    v = np.expand_dims(np.array(v), axis=0)
    if i == 0:
        ax_scales[1,i].plot(v.squeeze(0), color=vi_colours[0])
    else:
        ax_scales[1,i].plot(v.squeeze(0), color=vi_colours[0])
    ax_scales[0,i].set_title(k)  
    



ax_iters[1].set_title(f'K: {K}, Smoothed over {time_mean} iters')
ax_iters[0].set_ylabel('ELBO')
# ax_iters[0].set_ylim(-4000,-1400)
ax_iters[1].set_ylabel('Predictive LL')
ax_iters[2].set_ylabel('Non zero weights')
ax_iters[1].set_xlabel('Iters')
fig_iters.legend(loc='upper right')


fig_iters.tight_layout()
fig_iters.savefig(f'figures/bus_{K}_elbo_iters.png')

ax_scales[0,0].set_ylabel('Mean Scale')
ax_scales[1,0].set_ylabel('Mean Non zero weights')
fig_scales.legend(loc='upper right')
fig_scales.tight_layout()
fig_scales.savefig(f'figures/bus_{K}_scales_weights.png')