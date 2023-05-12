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
parser.add_argument('--num_samples',   '-s',   type=int,   nargs='?', default=25,  help="number of HMC samples to generate")
parser.add_argument('--num_tuning_samples',    type=int,   nargs='?', default=-1,  help="number of HMC tuning samples to generate (and discard)")
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

M = 6
J = 12
I = 50
Returns = 5

# def getPredLL(predictions, true_obs):
#     return(stats.nbnom(130, 1/(1+np.exp(-predictions["logits"]))).logpmf(true_obs).sum((-1,-2, -3)).mean(0))
    # return(stats.bernoulli(1/(1+np.exp(-predictions["logits"]))).logpmf(true_obs).sum((-1,-2)).mean(0))


for dataset_seed in args.dataset_seeds:
    covariates = {'weather': t.load('data/weather_train_{}.pt'.format(dataset_seed)).rename('plate_Years', 'plate_Birds', 'plate_Ids').float(),
        'quality': t.load('data/quality_train_{}.pt'.format(dataset_seed)).rename('plate_Years', 'plate_Birds', 'plate_Ids').float()}
    test_covariates = {'weather': t.load('data/weather_test_{}.pt'.format(dataset_seed)).rename('plate_Years', 'plate_Birds', 'plate_Ids').float(),
        'quality': t.load('data/quality_test_{}.pt'.format(dataset_seed)).rename('plate_Years', 'plate_Birds', 'plate_Ids').float()}
    all_covariates = {'weather': t.cat([covariates['weather'],test_covariates['weather']],-1),
        'quality': t.cat([covariates['quality'],test_covariates['quality']],-1)}


    if use_data:
        # Load data
        data = {'obs':t.load('data/birds_train_{}.pt'.format(dataset_seed)).float().rename('plate_Years', 'plate_Birds', 'plate_Ids','plate_Replicate')}
        # print(data)
        test_data = {'obs':t.load('data/birds_test_{}.pt'.format(dataset_seed)).float().rename('plate_Years', 'plate_Birds', 'plate_Ids','plate_Replicate')}
        all_data = {'obs': t.cat([data['obs'],test_data['obs']],-2)}


    else:
        # Generate data from model
        device = 'cpu'
        sizes = {'plate_Years': M, 'plate_Birds':J, 'plate_Ids':I, 'plate_Replicate': 5}

        def P(tr):
            '''
            Hierarchical Occupancy Model
            '''
            tr.sample('year_mean', alan.Normal(t.zeros(()), t.ones(())), plates='plate_Years')

            tr.sample('bird_mean', alan.Normal(tr['year_mean'], t.ones(())), plates='plate_Birds')

            tr.sample('beta', alan.Normal(tr['bird_mean'], t.ones(())), plates='plate_Ids')


            Phi = tr['beta']*tr['weather']
            #Presence of birds
            tr.sample('z', alan.Bernoulli(logits = Phi))


            tr.sample('alpha', alan.Normal(tr['bird_mean'], t.ones(())), plates='plate_Ids')
            p = tr['alpha']*tr['quality']

            #Observation of birds
            tr.sample('obs', alan.Bernoulli(logits=p*tr['z']), plates='plate_Replicate')

            sampledData = alan.sample(P, platesizes=sizes, covariates=covariates)
            data = {'obs': sampledData['obs']}

    # convert to numpy for pymc
    for c in (covariates, test_covariates, all_covariates):
        for x in c:
            c[x] = c[x].numpy()

    test_data = {'obs':t.load('data/birds_test_{}.pt'.format(dataset_seed)).float().rename('plate_Years', 'plate_Birds', 'plate_Ids','plate_Replicate')}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']],-2)}


    for d in (data, test_data, all_data):
        d['obs'] = d['obs'].numpy()


    alpha_means = np.zeros((num_runs, num_samples, M, J, I))
    train_times = np.zeros((num_runs, num_samples))

    p_lls = np.zeros((num_runs, num_samples))
    p_ll_times = np.zeros((num_runs, num_samples))

    for i in range(num_runs):
        model = pm.Model()

        with model:
            true_obs = pm.MutableData('true_obs', data['obs'])

            # Year level
            year_mean = pm.Normal('year_mean', mu=0, sigma=np.sqrt(1), shape=(M,))
            # mu_beta    = pm.Normal('mu_beta', mu=0, sigma=np.sqrt(0.0001))
            # beta       = pm.Normal('beta', mu=mu_beta, sigma=np.exp(sigma_beta), shape=(M,))

            # Bird level
            bird_mean = pm.Normal('bird_mean', mu=year_mean, sigma=1, shape=(M,J))
            # alpha       = pm.Normal('alpha', mu=beta, sigma=np.sqrt(np.exp(sigma_alpha)), shape=(M,J))

            # ID level
            beta = pm.Normal('beta', mu=bird_mean, sigma=1, shape=(M,J,I))
            alpha = pm.Normal('alpha', mu=bird_mean, sigma=1, shape=(M,J,I))

            # Covariates
            weather = pm.MutableData('weather', covariates['weather'])
            quality         = pm.MutableData('run_type', covariates['quality'])

            Phi = pm.Deterministic('Phi', beta*weather)
            p = pm.Deterministic('p', alpha*quality)

            z = pm.Bernoulli('z', p=1/(1+np.exp(-Phi)), shape=(M, J, I))

            zp = pm.Deterministic('zp', z*p)

            obs = pm.Bernoulli('obs', p=1/(1+np.exp(-zp)), shape=(M, J, I, Returns), observed=true_obs)

            p_ll = pm.Deterministic('p_ll', model.observedlogp)

            # breakpoint()
            var_names=['p_ll', 'obs','z','beta','alpha', 'bird_mean', 'year_mean']

            print("Sampling posterior WITH JAX!")
            trace = pmjax.sample_blackjax_nuts(draws=num_samples, tune=num_tuning_samples, chains=1, random_seed=i, target_accept=target_accept)#, random_seed=rng)#, discard_tuned_samples=False)
            train_times[i] = np.linspace(0,trace.attrs["sampling_time"],num_samples+num_tuning_samples+1)[num_tuning_samples+1:]

            alpha_means[i] = [trace.posterior.alpha[:,:j].mean(("chain", "draw")) for j in range(1, num_samples+1)]


            if use_data:
                pm.set_data({'weather': test_covariates['weather'],
                             'quality': test_covariates['quality'],
                             'true_obs': test_data['obs']})

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

        # variance in alpha estimates
        train_time_mean = np.mean(train_times,0).tolist()
        train_time_std_err  = (np.std(train_times,0)/np.sqrt(num_runs)).tolist()

        vars_mean = np.mean(np.std(alpha_means,0)**2, (-1, -2,-3)).tolist()

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

        mses_mean = ((alpha_means - sampledData['year_mean'].numpy())**2).mean((0,-1,-2,-3))

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
        with open(f"results/hmc_occupancy{args.results_tag}_p_ll_{dataset_seed}.json", 'w') as f:
            json.dump({"NUTS": p_lls}, f, indent=4)

        with open(f"results/hmc_occupancy{args.results_tag}_variance_{dataset_seed}.json", 'w') as f:
            json.dump({"NUTS": alpha_vars}, f, indent=4)
    else:
        with open(f"results/hmc_occupancy{args.results_tag}_MSE_{dataset_seed}.json", 'w') as f:
            json.dump({"NUTS": alpha_mses}, f, indent=4)

script_end_time = time.time()
print(f"Finished in {script_end_time - script_start_time}s.")

# breakpoint()
