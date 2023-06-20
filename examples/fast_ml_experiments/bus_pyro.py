import math
import pickle

import torch as t

import pyro
from pyro.distributions import Beta, Binomial, HalfCauchy, Normal, Pareto, Uniform, Bernoulli
from pyro.distributions.util import scalar_like
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.mcmc.util import initialize_model, summary
from pyro.util import ignore_experimental_warning

from bus_breakdown.bus_breakdown import generate_model as generate_ML
from alan.experiment_utils import seed_torch, n_mean


seed_torch(0)
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


M = 2
J = 3
I = 30

use_data = True

_, _, delays, covariates, _, _, _ = generate_ML(0,0, device, 2, 0, use_data)


delays = delays['obs'].rename(None).permute(2,0,1).unsqueeze(-1).float()

run_type = covariates['run_type'].rename(None).permute(1,0,2,3).float()
bus_company_name = covariates['bus_company_name'].rename(None).permute(1,0,2,3).float()

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


            # print(bus_company_name.shape)
            # print(phi.shape)
            # print(run_type.shape)
            # print(psi.shape)
            # print(alpha.view(*alpha.shape,1,1).shape)
            # print((bus_company_name @ phi + run_type @ psi).shape)
            logits = (alpha.view(*alpha.shape,1,1) + bus_company_name @ phi + run_type @ psi).permute(2,1,0,3)

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
    num_samples=1000,
    warmup_steps=5000,
    num_chains=7,
    initial_params=init_params,
    transforms=transforms,
)
mcmc.run(run_type, bus_company_name, delays)
samples = mcmc.get_samples()


with open(f'posteriors/bus_{use_data}.pkl', 'wb') as f:
    pickle.dump(samples, f)

sigma_beta_posterior_mean  = samples['sigma_beta'].mean(0)
mu_beta_posterior_mean = samples['mu_beta'].mean(0)

with open(f'posteriors/sigma_beta_mean_{use_data}.pkl', 'wb') as f:
    pickle.dump(sigma_beta_posterior_mean, f)

with open(f'posteriors/mu_beta_posterior_mean_{use_data}.pkl', 'wb') as f:
    pickle.dump(mu_beta_posterior_mean, f)