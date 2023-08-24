


fig.suptitle(f'K: {K}, Not Smoothed, Using Data: {use_data}')



ax[18,0].set_ylabel('ELBO')

ax[18,1].set_ylabel('Predictive LL')

ax[18,0].set_xlabel('Time')
ax[18,1].set_xlabel('Time')
ax[0,0].legend(loc='upper right')


plt.tight_layout()
plt.savefig(f'figures/movielens_test_data_{K}_{use_data}.png')