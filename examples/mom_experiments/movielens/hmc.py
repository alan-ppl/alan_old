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
parser.add_argument('--num_tuning_samples',    type=int,   nargs='?', default=1,  help="number of HMC tuning samples to generate (and discard)")
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

N, M = 20, 450
d_z = 18

def getPredLL(predictions, true_obs):
    return(stats.bernoulli(1/(1+np.exp(-predictions["logits"]))).logpmf(true_obs).sum((-1,-2)).mean(0))

for dataset_seed in args.dataset_seeds:
    # Load covariates
    covariates = {'x':t.load(f'data/weights_{N}_{M}_{dataset_seed}.pt')}
    test_covariates = {'x':t.load(f'data/test_weights_{N}_{M}_{dataset_seed}.pt')}
    all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
    covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)

    if use_data:
        # Load data
        data = {'obs':t.load(f'data/data_y_{N}_{M}_{dataset_seed}.pt')}
    else:
        # Generate data from model
        # device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        device = 'cpu'
        sizes = {'plate_1':M, 'plate_2':N}

        def P(tr):
            '''
            Heirarchical Model
            '''

            tr.sample('mu_z', alan.Normal(t.zeros((d_z,)).to(device),0.25*t.ones((d_z,)).to(device)))
            tr.sample('psi_z', alan.Normal(t.zeros((d_z,)).to(device), 0.25*t.ones((d_z,)).to(device)))

            tr.sample('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()), plates='plate_1')

            tr.sample('obs', alan.Bernoulli(logits = tr['z'] @ tr['x']))

        sampledData = alan.sample(P, varnames=('obs','z','mu_z','psi_z'), platesizes=sizes, covariates=covariates)
        data = {'obs': sampledData['obs']}

    # convert to numpy for pymc
    for c in (covariates, test_covariates, all_covariates):
        c['x'] = c['x'].numpy()

    test_data = {'obs':t.load(f'data/test_data_y_{N}_{M}_{dataset_seed}.pt')}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
    data['obs'] = data['obs'].rename('plate_1','plate_2')

    for d in (data, test_data, all_data):
        d['obs'] = d['obs'].numpy()


    z_means = np.zeros((num_runs, num_samples, M, d_z))
    train_times = np.zeros((num_runs, num_samples))

    p_lls = np.zeros((num_runs, num_samples))
    p_ll_times = np.zeros((num_runs, num_samples))

    for i in range(num_runs):
        model = pm.Model()

        with model:
            true_obs = pm.MutableData('true_obs', data['obs'])
            mu_z = pm.MvNormal('mu_z', mu=np.zeros(d_z), cov=0.25*np.eye(d_z))
            psi_z = pm.MvNormal('psi_z', mu=np.zeros(d_z), cov=0.25*np.eye(d_z))

            z = pm.MvNormal('z', mu=mu_z, cov=np.eye(d_z)*psi_z.exp(), shape=(M, d_z))

            x = pm.MutableData('x', covariates['x'])

            logits = pm.Deterministic('logits', (z @ x.transpose(0,2,1)).diagonal().transpose())
            # ^^ equivalent to:
            #       logits = pm.Deterministic('logits', np.einsum('ij,ikj->ik', z, x))
            # but pymc doesn't like einsums in Deterministic nodes
            
            obs = pm.Bernoulli('obs', logit_p = logits, observed=true_obs, shape=(M, N))

            p_ll = pm.Deterministic('p_ll', model.observedlogp)

            var_names = ["p_ll", "mu_z", "psi_z", "z", "logits", "obs"]

            print("Sampling posterior WITH JAX!")
            trace = pmjax.sample_blackjax_nuts(draws=num_samples, tune=num_tuning_samples, chains=1, random_seed=i, target_accept=target_accept)#, random_seed=rng)#, discard_tuned_samples=False)
            train_times[i] = np.linspace(0,trace.attrs["sampling_time"],num_samples+num_tuning_samples+1)[num_tuning_samples+1:]
            # train_times[i] = np.linspace(0,trace.attrs["sampling_time"],num_samples+1)[1:]

            z_means[i] = [trace.posterior.z[:,:j].mean(("chain", "draw")) for j in range(1, num_samples+1)]


            if use_data:
                pm.set_data({'x': test_covariates['x'], 'true_obs': test_data['obs']})

                print("Sampling posterior predictive WITH JAX!")
                if t.cuda.is_available():
                    t.cuda.synchronize()
                start = time.time()

                pp_trace = pm.sample_posterior_predictive(trace, var_names=var_names, random_seed=i, predictions=True)#, return_inferencedata=True)

                # test_ll = getPredLL(pp_trace.posterior_predictive, test_data['obs'])
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

        # variance in z estimates
        train_time_mean = np.mean(train_times,0).tolist()
        train_time_std_err  = (np.std(train_times,0)/np.sqrt(num_runs)).tolist()

        vars_mean = np.mean(np.std(z_means,0)**2, (-1, -2)).tolist()

        z_vars = {}
        for i in range(num_samples):
            z_vars[str(i+1)] = {
                "z": {
                    "mean_var": vars_mean[i],
                },
                "time_mean": train_time_mean[i],
                "time_std_err": train_time_std_err[i]
            }
        # breakpoint()
    else:
        # mse of z estimates (compared to true z)
        train_time_mean = np.mean(train_times,0).tolist()
        train_time_std_err  = (np.std(train_times,0)/np.sqrt(num_runs)).tolist()

        mses_mean = ((z_means - sampledData['z'].numpy())**2).mean((0,-1,-2))

        z_mses = {}
        for i in range(num_samples):
            z_mses[str(i+1)] = {
                "z": {
                    "mean_var": mses_mean[i],
                },
                "time_mean": train_time_mean[i],
                "time_std_err": train_time_std_err[i]
            }

    if use_data:
        with open(f"results/hmc_movielens{args.results_tag}_p_ll_{dataset_seed}.json", 'w') as f:
            json.dump({"NUTS": p_lls}, f, indent=4)

        with open(f"results/hmc_movielens{args.results_tag}_variance_{dataset_seed}.json", 'w') as f:
            json.dump({"NUTS": z_vars}, f, indent=4)
    else:
        with open(f"results/hmc_movielens{args.results_tag}_MSE_{dataset_seed}.json", 'w') as f:
            json.dump({"NUTS": z_mses}, f, indent=4) 

script_end_time = time.time()
print(f"Finished in {script_end_time - script_start_time}s.")

# breakpoint()