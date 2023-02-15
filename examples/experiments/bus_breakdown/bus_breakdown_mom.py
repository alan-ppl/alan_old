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

verbose = False

resultsFolder = "bus_breakdown/results"

device = t.device("cuda" if t.cuda.is_available() else "cpu")
device = "cpu"

M = 3
J = 3
I = 30

for useData in [True, False]:
    print(device)
    seed_torch(1)

    sizes = {'plate_Year': M, 'plate_Borough': J, 'plate_ID': I}
    
    P, Q, data, covariates, all_data, all_covariates = generate_model(M, J, I, device)

    if not useData:
        # Generate data
        sampledData = alan.sample(P, varnames=('obs','phi','psi','log_sigma_phi_psi', 'alpha', 'sigma_alpha', 'beta', 'mu_beta', 'sigma_beta'), platesizes=sizes, covariates=covariates)
        data = {'obs': sampledData['obs']}
        data['obs'] = data['obs'].rename('plate_Year', 'plate_Borough', 'plate_ID')

    # Make the model
    model = alan.Model(P, Q, data, covariates)
    model.to(device)

    Ks = {"tmc_new": [1,3,10,30], "global_k": [1,3,10,30,100,300,1000,3000,10000]}#,30000]}

    # "tmc_new" is the massively parallel approach 
    methods = ["tmc_new", "global_k"]

    if useData:
        elbos = {method: {k:[] for k in Ks[method]} for method in methods}
        elbo_times = {method: {k:[] for k in Ks[method]} for method in methods}

        p_lls = {method: {k:[] for k in Ks[method]} for method in methods}
        p_ll_times = {method: {k:[] for k in Ks[method]} for method in methods}

    expectations = {method: {k:[] for k in Ks[method]} for method in methods}
    expectation_times = {method: {k:[] for k in Ks[method]} for method in methods}
    
    # input("start?")

    for k in Ks["global_k"]:
        print(f"M={M}, J={J}, I={I}, k={k}")

        num_runs = 50#00
        for i in range(num_runs):
            if i % 5 == 0: print(i)

            if verbose: print("run", i)#, end=" ")

            validSamples = False
            while not validSamples:
                try:
                    if useData:
                        # Compute the elbos

                        if k in Ks["tmc_new"]:
                            start = time.time()
                            elbos["tmc_new"][k].append(model.elbo_tmc_new(k).item())
                            end = time.time()
                            elbo_times["tmc_new"][k].append(end-start)

                        start = time.time()
                        elbos["global_k"][k].append(model.elbo_global(k).item())
                        end = time.time()
                        elbo_times["global_k"][k].append(end-start)


                        # Compute the predictive log-likelihood
                        
                        for method in methods:
                            if method != "tmc_new" or k in Ks["tmc_new"]:
                                if verbose: print(method, end=". ")
                                error = True
                                while error:
                                    try:
                                        start = time.time()
                                        p_lls[method][k].append(model.predictive_ll(k, 100, data_all=all_data, covariates_all=all_covariates, sample_method=method)["obs"].item())
                                        end = time.time()

                                        if verbose: print(p_lls[method][k][-1])
                                        p_ll_times[method][k].append(end-start)

                                        error = False
                                    except ValueError:
                                        pass

                    # Compute (an estimate of) the expectation for each variable in the model
                    if k in Ks["tmc_new"]:
                        start = time.time()
                        expectations["tmc_new"][k].append(pp.mean(model.weights_tmc_new(k)))
                        end=time.time()
                        expectation_times["tmc_new"][k].append(end-start)

                    start = time.time()
                    expectations["global_k"][k].append(pp.mean(model.weights_global(k)))
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

                        if useData:
                            elbos[method][k] = elbos[method][k][:i]
                            elbo_times[method][k] = elbo_times[method][k][:i]

                            p_lls[method][k] = p_lls[method][k][:i]
                            p_ll_times[method][k] = p_ll_times[method][k][:i]

                        expectations[method][k] = expectations[method][k][:i]
                        expectation_times[method][k] = expectation_times[method][k][:i]

                    pass
                
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

                expectations[method][k] = {}
                expectations[method][k]["time_mean"] = float(np.mean(expectation_times[method][k]))
                expectations[method][k]["time_std_err"] = float(np.std(expectation_times[method][k]))
                for rv in rvs:
                    expectations[method][k][rv] = {"mean_var": mean_vars[rv]}

    if useData:
        file = f'{resultsFolder}/bus_breakdown_elbo_M{M}_J{J}_I{I}.json'
        with open(file, 'w') as f:
            json.dump(elbos, f)

        file = f'{resultsFolder}//bus_breakdown_p_ll_M{M}_J{J}_I{I}.json'
        with open(file, 'w') as f:
            json.dump(p_lls, f)

        file = f'{resultsFolder}/bus_breakdown_variance_M{M}_J{J}_I{I}.json'
        with open(file, 'w') as f:
            json.dump(expectations, f)
    else:
        file = f'{resultsFolder}/bus_breakdown_MSE_M{M}_J{J}_I{I}.json'
        with open(file, 'w') as f:
            json.dump(expectations, f)

    t.cuda.empty_cache()
    gc.collect()
