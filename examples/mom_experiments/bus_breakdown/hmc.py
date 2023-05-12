import arviz as az
import pymc as pm
# import pymc3 as pm
import numpy as np
import time
import torch as t
from scipy import stats
import alan
from alan.experiment_utils import seed_torch
import json
import sys
import argparse
from bus_breakdown import generate_model
import pymc.sampling.jax as pmjax
import jax

print(jax.default_backend())  # should print 'gpu'

import logging
logger = logging.getLogger('pymc')
logger.setLevel(logging.ERROR)

script_start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--use_data',      '-u',   default=False, action="store_true")
parser.add_argument('--verbose',       '-v',   default=False, action="store_true")
parser.add_argument('--num_runs',      '-n',   type=int,   nargs='?', default=1000,  help="number of runs")
parser.add_argument('--dataset_seeds', '-d',   type=int,   nargs='+', default=[0],  help="seeds for test/train split")
parser.add_argument('--results_tag',   '-t',   type=str,   nargs='?', default="",   help="string to attach to end of results filenames")
parser.add_argument('--num_samples',   '-s',   type=int,   nargs='?', default=10,  help="number of HMC samples to generate")
parser.add_argument('--num_tuning_samples',    type=int,   nargs='?', default=1,  help="number of HMC tuning samples to generate (and discard)--a value of -1 will set this equal to num_samples")
parser.add_argument('--target_accept', '-a',   type=float, nargs='?', default=0.8,  help="target acceptance rate for HMC")  

arglist = sys.argv[1:]
args = parser.parse_args(arglist)
print(args)

num_runs = args.num_runs
num_samples = args.num_samples
if args.num_tuning_samples == -1:
    num_tuning_samples = num_samples
else:
    num_tuning_samples = args.num_tuning_samples
use_data = args.use_data
target_accept = args.target_accept

# Initialize random number generator
# RANDOM_SEED = 0
# rng = np.random.default_rng(RANDOM_SEED)
# az.style.use("arviz-darkgrid")

seed_torch(0)

M, J, I = 3, 3, 30

# def getPredLL(predictions, true_obs):
#     return(stats.nbinom(130, 1/(1+np.exp(-predictions["logits"]))).logpmf(true_obs).sum((-1,-2, -3)).mean(0))
    # return(stats.bernoulli(1/(1+np.exp(-predictions["logits"]))).logpmf(true_obs).sum((-1,-2)).mean(0))


for dataset_seed in args.dataset_seeds:
    covariates = {'run_type': t.load(f'data/run_type_train_{dataset_seed}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...),
        'bus_company_name': t.load(f'data/bus_company_name_train_{dataset_seed}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
    test_covariates = {'run_type': t.load(f'data/run_type_test_{dataset_seed}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...),
        'bus_company_name': t.load(f'data/bus_company_name_test_{dataset_seed}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
    all_covariates = {'run_type': t.cat([covariates['run_type'],test_covariates['run_type']],-2),
        'bus_company_name': t.cat([covariates['bus_company_name'],test_covariates['bus_company_name']],-2)}

    bus_company_name_dim = covariates['bus_company_name'].shape[-1]
    run_type_dim = covariates['run_type'].shape[-1]

    if use_data:
        # Load data
        data = {'obs':t.load(f'data/delay_train_{dataset_seed}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
    else:
        # Generate data from model
        device = 'cpu'
        sizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':I}

        # def P(tr):
        #     '''
        #     Hierarchical Model
        #     '''
        #     #Year level
        #     tr.sample('sigma_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))#, plates = 'plate_Year') # GlobalVariance
        #     tr.sample('mu_beta', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))#, plates = 'plate_Year')    # GlobalMean
        #     tr.sample('beta', alan.Normal(tr['mu_beta'], tr['sigma_beta'].exp()), plates='plate_Year')                          # YearMean

        #     #Borough level
        #     tr.sample('sigma_alpha', alan.Normal(t.zeros(()).to(device), 0.25*t.ones(()).to(device)), plates = 'plate_Borough') # BoroughVariance
        #     tr.sample('alpha', alan.Normal(tr['beta'], tr['sigma_alpha'].exp()))                                                # BoroughMean

        #     #ID level
        #     tr.sample('log_sigma_phi_psi', alan.Normal(t.zeros(()).to(device), 0.0001*t.ones(()).to(device)))#, plates = 'plate_ID')          # WeightVariance
        #     tr.sample('psi', alan.Normal(t.zeros((run_type_dim,)).to(device), tr['log_sigma_phi_psi'].exp()))#, plates = 'plate_ID')          # J
        #     tr.sample('phi', alan.Normal(t.zeros((bus_company_name_dim,)).to(device), tr['log_sigma_phi_psi'].exp()))#, plates = 'plate_ID')  # C
        #     tr.sample('obs', alan.NegativeBinomial(total_count=130, logits=tr['alpha'] + tr['phi'] @ tr['bus_company_name'] + tr['psi'] @ tr['run_type']))  # Delay

        # sampledData = alan.sample(P, varnames=('obs','phi','psi','log_sigma_phi_psi', 'alpha', 'sigma_alpha', 'beta', 'mu_beta', 'sigma_beta'), platesizes=sizes, covariates=covariates)
        # data = {'obs': sampledData['obs']}    

        P, Q, data, covariates, all_data, all_covariates = generate_model(M, J, I, device, dataset_seed=dataset_seed)

        # Generate data
        sampledData = alan.sample(P, varnames=('obs','phi','psi','log_sigma_phi_psi', 'alpha', 'sigma_alpha', 'beta', 'mu_beta', 'sigma_beta'), platesizes=sizes, covariates=covariates)
        data = {'obs': sampledData['obs']}
        data['obs'] = data['obs'].rename('plate_Year', 'plate_Borough', 'plate_ID')

    # convert to numpy for pymc
    for c in (covariates, test_covariates, all_covariates):
        for x in c:
            c[x] = c[x].numpy()

    test_data = {'obs':t.load(f'data/delay_test_{dataset_seed}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']],-1)}

    for d in (data, test_data, all_data):
        d['obs'] = d['obs'].numpy()


    alpha_means = np.zeros((num_runs, num_samples, M, J))
    train_times = np.zeros((num_runs, num_samples))

    p_lls = np.zeros((num_runs, num_samples))
    p_ll_times = np.zeros((num_runs, num_samples))

    for i in range(num_runs):
        model = pm.Model()

        with model:
            true_obs = pm.MutableData('true_obs', data['obs'])

            # Year level
            sigma_beta = pm.Normal('sigma_beta', mu=0, sigma=np.sqrt(0.0001))
            mu_beta    = pm.Normal('mu_beta', mu=0, sigma=np.sqrt(0.0001))
            beta       = pm.Normal('beta', mu=mu_beta, sigma=np.exp(sigma_beta), shape=(M,))

            # Borough level
            sigma_alpha = pm.Normal('sigma_alpha', mu=0, sigma=np.sqrt(0.25), shape=(J,))
            alpha       = pm.Normal('alpha', mu=beta, sigma=np.sqrt(np.exp(sigma_alpha)), shape=(M,J))

            # ID level
            log_sigma_phi_psi = pm.Normal('log_sigma_phi_psi', mu=0, sigma=np.sqrt(0.0001))
            
            psi = pm.MvNormal('psi', mu=np.zeros(run_type_dim), cov=np.exp(log_sigma_phi_psi)*np.eye(run_type_dim), shape=(run_type_dim,))
            phi = pm.MvNormal('phi', mu=np.zeros(bus_company_name_dim), cov=np.exp(log_sigma_phi_psi)*np.eye(bus_company_name_dim), shape=(bus_company_name_dim,))
            
            # Covariates
            bus_company_name = pm.MutableData('bus_company_name', covariates['bus_company_name'])
            run_type         = pm.MutableData('run_type', covariates['run_type'])

            logits = pm.Deterministic('logits', (alpha + (phi @ bus_company_name.transpose(0,1,3,2) + psi @ run_type.transpose(0,1,3,2)).transpose(2,0,1)).transpose(1,2,0))

            obs = pm.NegativeBinomial('obs', n=130, p=1/(1+np.exp(-logits)), observed=true_obs, shape=(M, J, I))

            p_ll = pm.Deterministic('p_ll', model.observedlogp)
            
            # breakpoint()
            var_names=['p_ll', 'obs','logits','phi','psi','log_sigma_phi_psi', 'alpha', 'sigma_alpha', 'beta', 'mu_beta', 'sigma_beta']

            print("Sampling posterior WITH JAX!")
            trace = pmjax.sample_blackjax_nuts(draws=num_samples, tune=num_tuning_samples, chains=1, random_seed=i, target_accept=target_accept)#, random_seed=rng)#, discard_tuned_samples=False)
            train_times[i] = np.linspace(0,trace.attrs["sampling_time"],num_samples+num_tuning_samples+1)[num_tuning_samples+1:]
            # train_times[i,j] = trace.attrs["sampling_time"]

            alpha_means[i] = [trace.posterior.alpha[:,:j].mean(("chain", "draw")) for j in range(1, num_samples+1)]
            # alpha_means[i,j] = trace.posterior.alpha[0,-1]


            if use_data:
                # breakpoint()
                pm.set_data({'bus_company_name': test_covariates['bus_company_name'],
                            'run_type': test_covariates['run_type'],
                            'true_obs': test_data['obs']})

                print("Sampling posterior predictive WITH JAX!")
                if t.cuda.is_available():
                    t.cuda.synchronize()
                start = time.time()

                pp_trace = pm.sample_posterior_predictive(trace, var_names=var_names, random_seed=i, predictions=True)# discard_tuned_samples=False)#, return_inferencedata=True)

                test_ll = pp_trace.predictions.p_ll

                if t.cuda.is_available():
                    t.cuda.synchronize()
                end = time.time()

                p_ll_times[i] = np.linspace(0,end-start,num_samples+1)[1:] + train_times[i]

                p_lls[i] = test_ll

    if use_data:
        # predictive log-likelihood
        p_ll_mean = np.mean(p_lls,0).tolist()
        p_ll_std_err  = (np.std(p_lls,0)/np.sqrt(num_runs)).tolist()

        p_ll_time_mean = np.mean(p_ll_times,0).tolist()
        p_ll_time_std_err  = (np.std(p_ll_times,0)/np.sqrt(num_runs)).tolist()

        p_lls = {}
        for i in range(num_samples):
            p_lls[str(i+1)] = {
                "mean": p_ll_mean[i],
                "std_err": p_ll_std_err[i],
                "time_mean": p_ll_time_mean[i],
                "time_std_err": p_ll_time_std_err[i]
            }

        # variance in alpha estimates
        train_time_mean = np.mean(train_times,0).tolist()
        train_time_std_err  = (np.std(train_times,0)/np.sqrt(num_runs)).tolist()

        vars_mean = np.mean(np.std(alpha_means,0)**2, (-1, -2)).tolist()

        alpha_vars = {}
        for i in range(num_samples):
            alpha_vars[str(i+1)] = {
                "alpha": {
                    "mean_var": vars_mean[i],
                },
                "time_mean": train_time_mean[i],
                "time_std_err": train_time_std_err[i]
            }
        # breakpoint()
    else:
        # mse of alpha estimates (compared to true alpha)
        train_time_mean = np.mean(train_times,0).tolist()
        train_time_std_err  = (np.std(train_times,0)/np.sqrt(num_runs)).tolist()

        mses_mean = ((alpha_means - sampledData['alpha'].numpy())**2).mean((0,-1,-2))
        # breakpoint()

        alpha_mses = {}
        for i in range(num_samples):
            alpha_mses[str(i+1)] = {
                "alpha": {
                    "mean_var": mses_mean[i],
                },
                "time_mean": train_time_mean[i],
                "time_std_err": train_time_std_err[i]
            }

    if use_data:
        with open(f"results/hmc_bus_breakdown{args.results_tag}_p_ll_{dataset_seed}.json", 'w') as f:
            json.dump({"NUTS": p_lls}, f, indent=4)

        with open(f"results/hmc_bus_breakdown{args.results_tag}_variance_{dataset_seed}.json", 'w') as f:
            json.dump({"NUTS": alpha_vars}, f, indent=4)
    else:
        with open(f"results/hmc_bus_breakdown{args.results_tag}_MSE_{dataset_seed}.json", 'w') as f:
            json.dump({"NUTS": alpha_mses}, f, indent=4) 

script_end_time = time.time()
print(f"Finished in {script_end_time - script_start_time}s.")

# breakpoint()