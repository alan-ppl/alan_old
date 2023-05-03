import torch as t
import torch.nn as nn
import alan
import time
import numpy as np
import json
from alan.experiment_utils import seed_torch
import alan.postproc as pp
import gc
from bus_breakdown import generate_model
import sys

num_datasets = 4

nArgs = len(sys.argv)
verbose = False
forceCPU = False
num_runs = 1000

if nArgs == 1:
    pass
elif nArgs == 2:
    if sys.argv[1].isnumeric():
        num_runs = int(sys.argv[1])
    else:
        if sys.argv[1] in ("-v", "-c", "-vc", "-cv"):
            verbose = "v" in sys.argv[1]
            forceCPU = "c" in sys.argv[1] 
        else:
            raise ValueError("Non-numeric number of runs entered.\nUsage: python argtest.py [-vc] num_runs\n  -v:\tverbose output\n  -c:\t\tforce cpu use")
elif nArgs == 3:
    if sys.argv[2].isnumeric():
        num_runs = int(sys.argv[2])
    else:
        raise ValueError("Non-numeric number of runs entered.\nUsage: python argtest.py [-vc] num_runs\n  -v:\tverbose output\n  -c:\t\tforce cpu use")
    verbose = "v" in sys.argv[1]
    forceCPU = "c" in sys.argv[1]
else:
        raise ValueError("Too many arguments.\nUsage: python argtest.py [-vc] num_runs\n  -v:\tverbose output\n  -c:\t\tforce cpu use")

resultsFolder = "results"

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
print(device)

seed_torch(0)

M = 3
J = 3
I = 30
sizes = {'plate_Year': M, 'plate_Borough': J, 'plate_ID': I}

# "tmc_new" is the massively parallel approach 
methods = ["tmc_new", "global_k"]

Ks = {"tmc_new": [1,3,10,30], "global_k": [1,3,10,30,100,300,1000,3000,10000,30000,100000]}
Ks = {"tmc_new": [1,3,10], "global_k": [1,3,10,30,100,300,1000]}

for useData in [True, False]:
    expectationsPerDataset = []

    for datasetSeed in range(num_datasets):
        print(f"Dataset {datasetSeed+1}/{num_datasets}")

        P, Q, data, covariates, all_data, all_covariates = generate_model(M, J, I, device, AlanModule=True, run=datasetSeed)

        # Make the model
        model = alan.Model(P, Q())
        model.to(device)

        if not useData:
            # Generate data
            # sampledData = alan.sample(P, varnames=('obs','phi','psi','log_sigma_phi_psi', 'alpha', 'sigma_alpha', 'beta', 'mu_beta', 'sigma_beta'), platesizes=sizes, covariates=covariates)
            sampledData = model.sample_prior(platesizes = sizes, inputs = covariates, device=device)
            data = {'obs': sampledData['obs']}
            data['obs'] = data['obs'].rename('plate_Year', 'plate_Borough', 'plate_ID')


        if useData:
            elbos = {method: {k:[] for k in Ks[method]} for method in methods}
            elbo_times = {method: {k:[] for k in Ks[method]} for method in methods}

            p_lls = {method: {k:[] for k in Ks[method]} for method in methods}
            p_ll_times = {method: {k:[] for k in Ks[method]} for method in methods}

        expectations = {method: {k:[] for k in Ks[method]} for method in methods}
        expectation_times = {method: {k:[] for k in Ks[method]} for method in methods}
        
        for k in Ks["global_k"]:
            print(f"M={M}, J={J}, I={I}, k={k}")

            for i in range(num_runs):
                if verbose:
                    if i % 250 == 0: print(f"{i+1}/{num_runs}")

                validSamples = False
                while not validSamples:
                    try:
                        if useData:
                            # Compute the elbos

                            if k in Ks["tmc_new"]:
                                start = time.time()
                                sample = model.sample_perm(k, data=data, inputs=covariates, reparam=True, device=device)
                                elbos["tmc_new"][k].append(sample.elbo().item())
                                end = time.time()
                                elbo_times["tmc_new"][k].append(end-start)

                                error = True
                                while error:
                                    try:
                                        start = time.time()
                                        sample = model.sample_perm(k, data=data, inputs=covariates, reparam=False, device=device)
                                        pred_likelihood = model.predictive_ll(sample, N = 100, data_all=all_data, inputs_all=all_covariates)
                                        p_lls["tmc_new"][k].append(pred_likelihood["obs"].item())
                                        end = time.time()
                                        p_ll_times["tmc_new"][k].append(end-start)

                                        error = False
                                    except ValueError:
                                        pass

                                start = time.time()
                                sample = model.sample_perm(k, data=data, inputs=covariates, reparam=False, device=device)
                                expectations["tmc_new"][k].append(pp.mean(sample.weights()))
                                end=time.time()
                                expectation_times["tmc_new"][k].append(end-start)

                            start = time.time()
                            sample = model.sample_global(k, data=data, inputs=covariates, reparam=True, device=device)
                            elbos["global_k"][k].append(sample.elbo().item())
                            end = time.time()
                            elbo_times["global_k"][k].append(end-start)

                            error = True
                            while error:
                                try:
                                    start = time.time()
                                    sample = model.sample_global(k, data=data, inputs=covariates, reparam=False, device=device)
                                    pred_likelihood = model.predictive_ll(sample, N = 100, data_all=all_data, inputs_all=all_covariates)
                                    p_lls["global_k"][k].append(pred_likelihood["obs"].item())
                                    end = time.time()
                                    p_ll_times["global_k"][k].append(end-start)

                                    error = False
                                except ValueError:
                                    pass

                        # Compute (an estimate of) the expectation for each variable in the model
                        start = time.time()
                        sample = model.sample_global(k, data=data, inputs=covariates, reparam=False, device=device)
                        expectations["global_k"][k].append(pp.mean(sample.weights()))
                        end=time.time()
                        expectation_times["global_k"][k].append(end-start)


                        for method in methods:
                            if method != "tmc_new" or k in Ks["tmc_new"]:
                                # This is a very large model so the (log) probabilities get very small, meaning we must check for NaNs
                                assert not t.any(t.tensor([t.any(t.isnan(tensor)) for _, tensor in expectations[method][k][i].items()])).item()
                                if useData:
                                    assert not np.isnan(p_lls[method][k][i])
                                    assert not np.isnan(elbos[method][k][i])

                        validSamples = True

                    except AssertionError:
                        # print("AssertionError")

                        # Remove results generated by this run
                        for method in methods:
                            if method != "tmc_new" or k in Ks["tmc_new"]:
                                if useData:
                                    elbos[method][k] = elbos[method][k][:i]
                                    elbo_times[method][k] = elbo_times[method][k][:i]

                                    p_lls[method][k] = p_lls[method][k][:i]
                                    p_ll_times[method][k] = p_ll_times[method][k][:i]

                                expectations[method][k] = expectations[method][k][:i]
                                expectation_times[method][k] = expectation_times[method][k][:i]

                        pass
                    
                # input("Next run?")

            # Compute variance/MSE of results, store w/ mean/std_err execution time 
            for method in methods:
                if method != "tmc_new" or k in Ks["tmc_new"]:
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

            expectationsPerDataset.append(expectations.copy())

            # Clean up memory
            model.to("cpu")

            for x in [data, all_data, covariates, all_covariates]:
                for y in x.values():
                    y.to("cpu")
                    del y

            t.cuda.empty_cache()
            gc.collect()

    for k in Ks["global_k"]:
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
                    
                # rvs = list(expectations[vi_iter][0].keys())
                expectations[method][k] = {}
                expectations[method][k]["time_mean"] = np.mean([expectationsPerDataset[d][method][k]["time_mean"] for d in range(num_datasets)])
                expectations[method][k]["time_std_err"] = np.mean([expectationsPerDataset[d][method][k]["time_std_err"] for d in range(num_datasets)])
                for rv in rvs:
                    expectations[method][k][rv] = {"mean_var": np.mean([expectationsPerDataset[d][method][k][rv]["mean_var"] for d in range(num_datasets)])}
            

    if useData:
        file = f'{resultsFolder}/bus_breakdown_elbo_M{M}_J{J}_I{I}.json'
        with open(file, 'w') as f:
            json.dump(elbos, f, indent=4)

        file = f'{resultsFolder}/bus_breakdown_p_ll_M{M}_J{J}_I{I}.json'
        with open(file, 'w') as f:
            json.dump(p_lls, f, indent=4)

        file = f'{resultsFolder}/bus_breakdown_variance_M{M}_J{J}_I{I}.json'
        with open(file, 'w') as f:
            json.dump(expectations, f, indent=4)
    else:
        file = f'{resultsFolder}/bus_breakdown_MSE_M{M}_J{J}_I{I}.json'
        with open(file, 'w') as f:
            json.dump(expectations, f, indent=4)

    print(f"Finished useData={useData}")

