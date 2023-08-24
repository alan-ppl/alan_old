import pickle
import matplotlib.pyplot as plt
import numpy as np



for use_data in [False, True]:
    fig, ax = plt.subplots(4,1, figsize=(7, 9.0))
    with open(f'posteriors/bus_{use_data}.pkl', 'rb') as f:
        bus_samples  = pickle.load(f)


    sigma_beta_posterior_mean  = bus_samples['sigma_beta'].reshape(7,1000).mean(0)
    # sigma_beta_posterior_std  = samples['sigma_beta'].reshape(7,1000).std(0)
    mu_beta_posterior_mean = bus_samples['mu_beta'].reshape(7,1000).mean(0)
    # mu_beta_posterior_std = samples['mu_beta'].reshape(7,1000).std(0)

    length = sigma_beta_posterior_mean.shape[0]
    ax[0].set_title('Bus')
    ax[0].plot(np.arange(length), sigma_beta_posterior_mean, c='red')
    ax[1].plot(np.arange(length), mu_beta_posterior_mean)
    ax[0].set_ylabel('sigma_beta')


    ax[1].set_ylabel('mu_beta')

    with open(f'posteriors/movielens_{use_data}.pkl', 'rb') as f:
        movielens_samples  = pickle.load(f)


    mu_z_posterior_mean  = movielens_samples['mu_z'].sum(-1).reshape(7,1000,18).std(0)[:,0]
    psi_z_posterior_mean = movielens_samples['psi_z'].sum(-1).reshape(7,1000,18).std(0)[:,0]


    length = mu_z_posterior_mean.shape[0]

    pos = ax[2].get_position()
    new_pos = [pos.x0, pos.y0-0.03, pos.width, pos.height]
    ax[2].set_position(new_pos)

    ax[2].set_title('Movielens')
    ax[2].plot(np.arange(length), mu_z_posterior_mean, c='green')

    pos = ax[3].get_position()
    new_pos = [pos.x0, pos.y0-0.03, pos.width, pos.height]
    ax[3].set_position(new_pos)

    ax[3].plot(np.arange(length), psi_z_posterior_mean,c='orange')

    ax[2].set_ylabel('mu_z')


    ax[3].set_ylabel('psi_z')
    plt.savefig(f'figures/NUTS_{use_data}.png')


