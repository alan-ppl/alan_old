import pickle
import matplotlib.pyplot as plt
import numpy as np



for use_data in [True]:
    fig_mu, ax_mu = plt.subplots(18,1, figsize=(7, 5*8.0))
    fig_psi, ax_psi = plt.subplots(18,1, figsize=(7, 5*8.0))
    with open(f'posteriors/movielens_{use_data}.pkl', 'rb') as f:
        movielens_samples  = pickle.load(f)

    for i in range(18):
        mu_z_posterior_mean  = movielens_samples['mu_z'].sum(-1).reshape(7,1000,18).std(0)[:,i]
        


        length = mu_z_posterior_mean.shape[0]

        ax_mu[i].plot(np.arange(length), mu_z_posterior_mean, c='green')
        ax_mu[i].set_ylabel(f'mu_z_{i}')


        psi_z_posterior_mean = movielens_samples['psi_z'].sum(-1).reshape(7,1000,18).std(0)[:,i]        
        length = psi_z_posterior_mean.shape[0]
        
        ax_psi[i].plot(np.arange(length), psi_z_posterior_mean,c='orange')

        ax_psi[i].set_ylabel(f'psi_z_{i}')

    fig_mu.tight_layout()   
    fig_mu.savefig(f'figures/NUTS_movielns_mu_{use_data}.png')
    fig_psi.tight_layout()  
    fig_psi.savefig(f'figures/NUTS_movielns_psi_{use_data}.png')
