import arviz as az
import pymc as pm
# import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import torch as t
from scipy import stats
import alan
import json

# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu0')
# import jax
# jax.default.backend()

# device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# %config InlineBackend.figure_format = 'retina'
# Initialize random number generator
RANDOM_SEED = 145
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

num_runs = 3

N, M = 20, 450
useData = False
d_z = 18

# Load covariates
covariates = {'x':t.load('data/weights_{0}_{1}.pt'.format(N,M))}
test_covariates = {'x':t.load('data/test_weights_{0}_{1}.pt'.format(N,M))}
all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)


if useData:
    # Load data
    data = {'obs':t.load('data/data_y_{0}_{1}.pt'.format(N, M))}
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

for c in (covariates, test_covariates, all_covariates):
    c['x'] = c['x'].numpy()

test_data = {'obs':t.load('data/test_data_y_{0}_{1}.pt'.format(N, M))}
all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
data['obs'] = data['obs'].rename('plate_1','plate_2')

for d in (data, test_data, all_data):
    d['obs'] = d['obs'].numpy()

# print(all_covariates)
# print(all_data)
# print(all_covariates['x'].shape)
# print(all_data['obs'].shape)

# breakpoint()

def getPredLL(predictions, true_obs):
    ll_mu_z = stats.norm(0,0.25).logpdf(predictions["mu_z"]).sum(2)  # sum over dim 2 bc elements are iid
    ll_psi_z = stats.norm(0,0.25).logpdf(predictions["psi_z"]).sum(2)

    ll_z = stats.norm(predictions["mu_z"], np.exp(np.asarray(predictions["psi_z"]))).logpdf(predictions["z"]).sum(2)

    # ll_obs = stats.bernoulli(1/(1+np.exp(-predictions["logits"]))).logpmf(predictions["obs"]).sum((2,3))
    # ll_obs = stats.bernoulli(true_obs).logpmf(predictions["obs"]).sum((2,3)) # <- nonsense
    ll_obs = stats.bernoulli(1/(1+np.exp(-predictions["logits"]))).logpmf(true_obs).sum((2,3))


    ll_total = (ll_mu_z + ll_psi_z + ll_z + ll_obs).mean()
    return ll_total

z_vars = []
z_var_times = []
z_means = []
elbos = []
p_lls = []
p_ll_times = []

for i in range(num_runs):
    model = pm.Model()

    with model:
        true_obs = pm.MutableData('true_obs', data['obs'])
        mu_z = pm.MvNormal('mu_z', mu=np.zeros(d_z), cov=0.25*np.eye(d_z))
        psi_z = pm.MvNormal('psi_z', mu=np.zeros(d_z), cov=0.25*np.eye(d_z))

        z = pm.MvNormal('z', mu=mu_z, cov=np.eye(d_z)*psi_z.exp(), shape=(M,))

        x = pm.MutableData('x', covariates['x']).transpose(0,2,1)

        logits = pm.Deterministic('logits', z @ x)
        
        # breakpoint()
        obs = pm.Bernoulli('obs', logit_p = logits, observed=true_obs)

        var_names = ["mu_z", "psi_z", "z", "logits", "obs"]

        # start = time.time()
        # idata = pm.sample(draws=1, tune=100, random_seed=rng)
        # print(time.time() - start)

        # get z mean over k samples

        start = time.time()
        idata = pm.sample_posterior_predictive(
            # idata,
            pm.sample(draws=25, tune=25, random_seed=rng),
            var_names=var_names,
            return_inferencedata=True,
            predictions=True,
            extend_inferencedata=True,
            random_seed=rng,
            progressbar=False
        )
        end = time.time()
        z_var_times.append(end-start)
        z_means.append(idata.predictions["z"].mean(("chain", "draw")))
        if useData:
            z_vars.append((idata.predictions["z"].std(("chain", "draw"))**2).mean())

        # start = time.time()
        # idata = pm.sample_posterior_predictive(
        #     # idata,
        #     pm.sample(draws=25, tune=25, random_seed=rng),
        #     var_names=var_names,
        #     return_inferencedata=True,
        #     predictions=True,
        #     extend_inferencedata=True,
        #     random_seed=rng,
        #     progressbar=False
        # )
        # end = time.time()
        train_time = end - start
        train_ll = getPredLL(idata.predictions, test_data['obs'])

        # Now get predictive log likelihood    
        # pm.set_data({'x': test_covariates['x'], 'true_obs': test_data['obs']})
        pm.set_data({'x': all_covariates['x'], 'true_obs': all_data['obs']})


        start = time.time()
        idata = pm.sample_posterior_predictive(
            # idata,
            pm.sample(draws=25, tune=25, random_seed=rng),
            var_names=var_names,
            return_inferencedata=True,
            predictions=True,
            extend_inferencedata=True,
            random_seed=rng,
            progressbar=False
        )

        ll_total = getPredLL(idata.predictions, all_data['obs']) - train_ll
        end = time.time()
        
        p_lls.append(ll_total)
        p_ll_times.append(end-start)
        # breakpoint()

print("mean p_ll: ", np.mean(p_lls))
print("mean p_ll time: ", np.mean(p_ll_times))

if useData:
    print("mean z var: ", np.mean(z_vars))
    print("mean z var time: ", np.mean(z_var_times))

    results =  {"p_ll": np.mean(p_lls),
                "p_ll_time": np.mean(p_ll_times),
                "p_ll_std": np.std(p_lls),
                "p_ll_time_std": np.std(p_ll_times),
                
                "z_var": np.mean(z_vars),
                "z_var_time": np.mean(z_var_times),
                "z_var_std": np.std(z_vars),
                "z_var_time_std": np.std(z_var_times)}
    
    # breakpoint()
else:
    # breakpoint()
    # squared_error = np.mean([((sampledData["z"].mean(0) - z_est)**2) for z_est in z_means])
    mse = np.mean([np.mean([((sampledData["z"][j] - np.asarray(z_est))**2).numpy() for z_est in z_means]) for j in range(M)])
    print("mse: ", mse)

    results =  {"p_ll": np.mean(p_lls),
                "p_ll_time": np.mean(p_ll_times),
                "p_ll_std": np.std(p_lls),
                "p_ll_time_std": np.std(p_ll_times),
                
                "z_mse": mse,
                "z_var_time": np.mean(z_var_times),
                "z_var_time_std": np.std(z_var_times)}

with open(f"results/hmc_result{'' if useData else '_mse'}.json", 'w') as f:
            json.dump(results, f)

breakpoint()