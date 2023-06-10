import math
import pickle


import pandas as pd
import torch as t

import pyro
from pyro.distributions import Beta, Binomial, HalfCauchy, Normal, Pareto, Uniform, Bernoulli
from pyro.distributions.util import scalar_like
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.mcmc.util import initialize_model, summary
from pyro.util import ignore_experimental_warning

from movielens.movielens import generate_model as generate_ML
from alan.experiment_utils import seed_torch, n_mean


seed_torch(0)
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


M=300
N=5


def movielens(x, ratings):
    r"""
    """

    mu_z_prior = Normal(t.zeros(18,1), t.ones(18,1))

    mu_z = pyro.sample("mu_z", mu_z_prior)
    
    psi_z_prior = Normal(t.zeros(18,1), t.ones(18,1))
    psi_z = pyro.sample("psi_z", psi_z_prior)


    with pyro.plate("plate_1", M):
        z = pyro.sample("z", Normal(mu_z, psi_z.exp()))
        return pyro.sample("obs", Bernoulli(logits=(z.T.unsqueeze(1) * x).sum(-1).T), obs=ratings)

_, _, ratings, x, _, _, _ = generate_ML(N,M, device, 2, 0, True)


ratings = ratings['obs'].rename(None).T
x = x['x'].rename(None)


init_params, potential_fn, transforms, _ = initialize_model(
        movielens,
        model_args=(x, ratings),
        num_chains=7,
        jit_compile=True,
        skip_jit_warnings=True,
    )
nuts_kernel = NUTS(potential_fn=potential_fn)
mcmc = MCMC(
    nuts_kernel,
    num_samples=1000,
    warmup_steps=1000,
    num_chains=7,
    initial_params=init_params,
    transforms=transforms,
)
mcmc.run(x, ratings)
samples = mcmc.get_samples()


mu_z_posterior_mean  = samples['mu_z'].sum(-1).mean(0)[0]
psi_z_posterior_mean = samples['psi_z'].sum(-1).mean(0)[0]

with open('mu_z_posterior_mean.pkl', 'wb') as f:
    pickle.dump(mu_z_posterior_mean, f)

with open('psi_z_posterior_mean.pkl', 'wb') as f:
    pickle.dump(psi_z_posterior_mean, f)