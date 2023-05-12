import torch as t
import torch.nn as nn
import alan
import time
import numpy as np
import json
from alan.experiment_utils import seed_torch
import alan.postproc as pp
import gc
import sys
import argparse

if t.cuda.is_available():
    t.cuda.synchronize()
script_start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cpu',           '-c',   type=bool,  nargs='?', default=False)
parser.add_argument('--verbose',       '-v',   type=bool,  nargs='?', default=False)
parser.add_argument('--num_runs',      '-n',   type=int,   nargs='?', default=1000,  help="number of runs")
parser.add_argument('--dataset_seeds', '-d',   type=int,   nargs='+', default=[0],  help="seeds for test/train split")
parser.add_argument('--results_tag',   '-t',   type=str,   nargs='?', default="",   help="string to attach to end of results filenames")

arglist = sys.argv[1:]
args = parser.parse_args(arglist)
print(args)

forceCPU = args.cpu
verbose = args.verbose
num_runs = args.num_runs
dataset_seeds = args.dataset_seeds
results_tag = args.results_tag

resultsFolder = "results"

device = "cpu" if forceCPU else t.device("cuda" if t.cuda.is_available() else "cpu")
print(device)


M, N = 450, 20
sizes = {'plate_1':M, 'plate_2':N}
d_z = 18

# "tmc_new" is the massively parallel approach 
methods = ["tmc_new", "global_k"]

Ks = {"tmc_new": [1,3,10,30,100], "global_k": [1,3,10,30,100,300,1000,3000,10000,30000,100000]}#,300000]}


def P(tr):
    '''
    Heirarchical Model
    '''

    tr.sample('mu_z', alan.Normal(t.zeros((d_z,)).to(device),0.25*t.ones((d_z,)).to(device)))
    tr.sample('psi_z', alan.Normal(t.zeros((d_z,)).to(device), 0.25*t.ones((d_z,)).to(device)))

    tr.sample('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()), plates='plate_1')

    tr.sample('obs', alan.Bernoulli(logits = tr['z'] @ tr['x']))

def Q(tr):
    '''
    Heirarchical Model
    '''

    tr.sample('mu_z', alan.Normal(t.zeros((d_z,)).to(device), 0.25*t.ones((d_z,)).to(device)))
    tr.sample('psi_z', alan.Normal(t.zeros((d_z,)).to(device), 0.25*t.ones((d_z,)).to(device)))

    tr.sample('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()), plates='plate_1')

    # tr.sample('obs', alan.Bernoulli(logits = tr['z'] @ tr['x']))

for dataset_seed in dataset_seeds:
    print(f"Dataset seed: {dataset_seed}")

    # Run the experiment
    for useData in [True, False]:
        seed_torch(dataset_seed)

        ## Load covariates
        covariates = {'x':t.load(f'data/weights_{N}_{M}_{dataset_seed}.pt').to(device)}
        test_covariates = {'x':t.load(f'data/test_weights_{N}_{M}_{dataset_seed}.pt').to(device)}
        all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
        covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)

        if useData:
            # Load data
            data = {'obs':t.load(f'data/data_y_{N}_{M}_{dataset_seed}.pt').to(device)}
            test_data = {'obs':t.load(f'data/test_data_y_{N}_{M}_{dataset_seed}.pt').to(device)}
            all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
            data['obs'] = data['obs'].rename('plate_1','plate_2')
        else:
            # Generate data
            sampledData = alan.sample(P, varnames=('obs','z','mu_z','psi_z'), platesizes=sizes, covariates=covariates)

            data = {'obs': sampledData['obs']}
            test_data = {'obs':t.load(f'data/test_data_y_{N}_{M}_{dataset_seed}.pt').to(device)}
            all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
            data['obs'] = data['obs'].rename('plate_1','plate_2')

        # Make the model 
        model = alan.Model(P, Q, data, covariates)
        model.to(device)

        if useData:
            elbos = {method: {k:[] for k in Ks[method]} for method in methods}
            elbo_times = {method: {k:[] for k in Ks[method]} for method in methods}

            p_lls = {method: {k:[] for k in Ks[method]} for method in methods}
            p_ll_times = {method: {k:[] for k in Ks[method]} for method in methods}

        expectations = {method: {k:[] for k in Ks[method]} for method in methods}
        expectation_times = {method: {k:[] for k in Ks[method]} for method in methods}

        for k in Ks["global_k"]:
            # print(f"M={M}, N={N}, k={k}")

            for i in range(num_runs):
                if verbose: 
                    if i % 250 == 0: print(f"{i+1}/{num_runs}")

                if useData:
                    # Compute the elbos

                    if k in Ks["tmc_new"]:
                        if t.cuda.is_available():
                            t.cuda.synchronize()
                        start = time.time()

                        elbos["tmc_new"][k].append(model.elbo_tmc_new(k).item())

                        if t.cuda.is_available():
                            t.cuda.synchronize()
                        end = time.time()

                        elbo_times["tmc_new"][k].append(end-start)

                    if t.cuda.is_available():
                        t.cuda.synchronize()
                    start = time.time()

                    elbos["global_k"][k].append(model.elbo_global(k).item())

                    if t.cuda.is_available():
                        t.cuda.synchronize()
                    end = time.time()
                    
                    elbo_times["global_k"][k].append(end-start)

                    # Compute the predictive log-likelihood
                    for method in methods:
                        if method != "tmc_new" or k in Ks["tmc_new"]:
                            error = True
                            while error:
                                try:
                                    if t.cuda.is_available():
                                        t.cuda.synchronize()
                                    start = time.time()

                                    p_lls[method][k].append(model.predictive_ll(k, 100, data_all=all_data, covariates_all=all_covariates, sample_method=method)["obs"].item())

                                    if t.cuda.is_available():
                                        t.cuda.synchronize()
                                    end = time.time()

                                    p_ll_times[method][k].append(end-start)

                                    error = False
                                except ValueError:
                                    print("NaN p_ll!")
                                    pass

                # Compute (an estimate of) the expectation for each variable in the model
                if t.cuda.is_available():
                    t.cuda.synchronize()
                start = time.time()

                expectations["global_k"][k].append(pp.mean(model.weights_global(k)))
                
                if t.cuda.is_available():
                    t.cuda.synchronize()
                end=time.time()

                expectation_times["global_k"][k].append(end-start)

                if k in Ks["tmc_new"]:
                    if t.cuda.is_available():
                        t.cuda.synchronize()
                    start = time.time()

                    expectations["tmc_new"][k].append(pp.mean(model.weights_tmc_new(k)))
                    
                    if t.cuda.is_available():
                        t.cuda.synchronize()
                    end=time.time()

                    expectation_times["tmc_new"][k].append(end-start)

                # input("Next run?")

            # Compute mean/std_err/variance/MSE of results, store w/ mean/std_err execution time 
            for method in methods:
                if method != "tmc_new" or k in Ks["tmc_new"]:
                    if useData:
                        elbos[method][k] = {'mean': np.mean(elbos[method][k]),
                                            'std_err': np.std(elbos[method][k])/np.sqrt(num_runs),
                                            'time_mean': np.mean(elbo_times[method][k]),
                                            'time_std_err': np.std(elbo_times[method][k])/np.sqrt(num_runs)}

                    
                        p_lls[method][k] = {'mean': np.mean(p_lls[method][k]),
                                            'std_err': np.std(p_lls[method][k])/np.sqrt(num_runs),
                                            'time_mean': np.mean(p_ll_times[method][k]),
                                            'time_std_err': np.std(p_ll_times[method][k])/np.sqrt(num_runs)}


                    rvs = list(expectations[method][k][0].keys())
                    mean_vars = {rv: [] for rv in rvs}  # average element variance for each rv

                    if useData:
                        expectation_means = {rv: sum([x[rv] for x in expectations[method][k]])/num_runs for rv in rvs}
                    else:
                        expectation_means = {rv: sampledData[rv] for rv in rvs}  # use the true values for the sampled data
                        
                    sq_errs = {rv: [] for rv in rvs}

                    for est in expectations[method][k]:
                        for rv in est:
                            sq_err = ((expectation_means[rv] - est[rv])**2).cpu()
                            sq_errs[rv].append(sq_err.rename(None))
                    
                    for rv in rvs:
                        mean_vars[rv] = float(t.mean(t.stack(sq_errs[rv])))

                    for run in expectations[method][k]:
                        for rv in rvs:
                            run[rv].to("cpu")
                            del run[rv]

                    expectations[method][k] = {}
                    expectations[method][k]["time_mean"] = float(np.mean(expectation_times[method][k]))
                    expectations[method][k]["time_std_err"] = float(np.std(expectation_times[method][k]))
                    for rv in rvs:
                        expectations[method][k][rv] = {"mean_var": mean_vars[rv]}

        # Clean up memory
        model = model.to("cpu")

        for x in [data, test_data, all_data, covariates, test_covariates, all_covariates]:
            for y in x:
                x[y] = x[y].to("cpu")
            del x

        t.cuda.empty_cache()
        gc.collect()

        if useData:
            file = f'{resultsFolder}/movielens{results_tag}_elbo_{dataset_seed}.json'
            with open(file, 'w') as f:
                json.dump(elbos, f, indent=4)

            file = f'{resultsFolder}/movielens{results_tag}_p_ll_{dataset_seed}.json'
            with open(file, 'w') as f:
                json.dump(p_lls, f, indent=4)

            file = f'{resultsFolder}/movielens{results_tag}_variance_{dataset_seed}.json'
            with open(file, 'w') as f:
                json.dump(expectations, f, indent=4)
        else:
            file = f'{resultsFolder}/movielens{results_tag}_MSE_{dataset_seed}.json'
            with open(file, 'w') as f:
                json.dump(expectations, f, indent=4)

        print(f"Finished useData={useData}")

if t.cuda.is_available():
    t.cuda.synchronize()
script_end_time = time.time()
print(f"Finished. Took {script_end_time - script_start_time}s.")