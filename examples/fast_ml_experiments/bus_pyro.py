import pickle


import torch as t

import pyro
from pyro.distributions import Binomial, Normal
from pyro.infer import MCMC, NUTS
from pyro.infer.mcmc.util import initialize_model

from bus_breakdown.bus_breakdown import generate_model as generate_ML
from alan.experiment_utils import seed_torch, n_mean


seed_torch(0)
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


M = 3
J = 3
I = 60

use_data = False

_, _, delays, covariates, _, _, _ = generate_ML(0,0, device, 2, 0, use_data)


delays = delays['obs'].rename(None).permute(2,1,0).unsqueeze(-1).float()

run_type = covariates['run_type'].rename(None).float()
print(run_type.shape)
bus_company_name = covariates['bus_company_name'].rename(None).float()

bus_company_name_dim = covariates['bus_company_name'].shape[-1]
run_type_dim = covariates['run_type'].shape[-1]

def bus(run_type, bus_company_name, delays):
    r"""
    """

    sigma_beta = pyro.sample("sigma_beta", Normal(0,1))
    
    mu_beta = pyro.sample("mu_beta", Normal(0,1))



    log_sigma_phi_psi = pyro.sample("log_sigma_phi_psi", Normal(0,1))
    psi = pyro.sample("psi", Normal(t.zeros(run_type_dim,1),log_sigma_phi_psi.exp()))
    phi = pyro.sample("phi", Normal(t.zeros((bus_company_name_dim,1)),log_sigma_phi_psi.exp()))

    with pyro.plate("plate_Year", M):
        beta = pyro.sample("beta", Normal(mu_beta, sigma_beta.exp()))

        with pyro.plate("plate_borough", J):
            sigma_alpha = pyro.sample("sigma_alpha", Normal(0, 1))
            alpha = pyro.sample("alpha", Normal(beta, sigma_alpha.exp()))



            logits = (alpha.view(*alpha.shape,1,1) + bus_company_name @ phi + run_type @ psi).permute(2,1,0,3)
            print(logits.shape)
            return pyro.sample("delays", Binomial(total_count=131, logits=logits), obs=delays)

 


init_params, potential_fn, transforms, _ = initialize_model(
        bus,
        model_args=(run_type, bus_company_name, delays),
        num_chains=7,
        jit_compile=True,
        skip_jit_warnings=True,
    )
nuts_kernel = NUTS(potential_fn=potential_fn)
mcmc = MCMC(
    nuts_kernel,
    num_samples=2,
    warmup_steps=1,
    num_chains=7,
    initial_params=init_params,
    transforms=transforms,
)
mcmc.run(run_type, bus_company_name, delays)
samples = mcmc.get_samples()


with with open(f'posteriors/bus_{use_data}.pkl', 'wb') as f:
    pickle.dump(samples, f)

sigma_beta_posterior_mean  = samples['sigma_beta'].mean(0)
mu_beta_posterior_mean = samples['mu_beta'].mean(0)

with open(f'posteriors/sigma_beta_mean_{use_data}.pkl', 'wb') as f:
    pickle.dump(sigma_beta_posterior_mean, f)

with open(f'posteriors/mu_beta_posterior_mean_{use_data}.pkl', 'wb') as f:
    pickle.dump(mu_beta_posterior_mean, f)