import torch as t
import torch.nn as nn
import alan
import time
import numpy as np
import json
from alan.experiment_utils import seed_torch
import alan.postproc as pp
import gc
from potus import generate_model
import sys


num_datasets = 3

num_vi_iters = 10#0
vi_iter_step = 1


num_runs = 250
nArgs = len(sys.argv)
verbose = False
forceCPU = False


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

device = "cpu" if forceCPU else t.device("cuda:0" if t.cuda.is_available() else "cpu")
print(device)

seed_torch(0)

M = 3
J = 3
I = 30
sizes = {'plate_Year': M, 'plate_Borough': J, 'plate_ID': I}

k = 1
learningRates = [0.5, 0.25]#,0.1, 0.01, 0.001]#, 0.0001]

# Run the experiment
for lr in learningRates:
    for useData in [False, True]:

        expectationsPerDataset = []

        for datasetSeed in range(num_datasets):

            P, Q, data, covariates, all_data, all_covariates = generate_model(M, J, I, device, AlanModule=True, run=datasetSeed)
            # if not useData:
            #     # Generate data
            #     tempModel = alan.Model(P, Q())
            #     tempModel.to(device)

            #     sampledData = tempModel.sample_prior(platesizes = sizes, inputs = covariates, device=device)
            #     data = {'obs': sampledData['obs']}
            #     data['obs'] = data['obs'].rename('plate_Year', 'plate_Borough', 'plate_ID')

            vi_iter_counts = [x for x in range(0, num_vi_iters+1, vi_iter_step)]

            if useData:
                elbos = {count :[] for count in vi_iter_counts}
                elbo_times = {count:[] for count in vi_iter_counts}

                p_lls = {count:[] for count in vi_iter_counts}
                p_ll_times = {count:[] for count in vi_iter_counts}

            expectations = {count:[] for count in vi_iter_counts}
            expectation_times = {count:[] for count in vi_iter_counts}

            for i in range(num_runs):
                # Make the model
                model = alan.Model(P, Q())

                model.to(device)

                if not useData:
                    # Generate data
                    sampledData = model.sample_prior(platesizes = sizes, inputs = covariates, device=device)
                    data = {'obs': sampledData['obs']}
                    data['obs'] = data['obs'].rename('plate_Year', 'plate_Borough', 'plate_ID')


                opt = t.optim.Adam(model.parameters(), lr=lr)

                train_time = 0

                if verbose:
                    if i % 100 == 0: print(f"{i+1}/{num_runs}")

                for vi_iter in range(num_vi_iters+1):

                    if vi_iter % vi_iter_step == 0:
                        # if verbose: print(f"{vi_iter}/{num_vi_iters}")

                        validSamples = False
                        while not validSamples:
                            try:
                                if useData:
                                    # Compute the elbo
                                    start = time.time()
                                    sample = model.sample_perm(k, data=data, inputs=covariates, reparam=True, device=device)
                                    elbo = sample.elbo()
                                    elbos[vi_iter].append(elbo.item())
                                    end = time.time()
                                    elboTime = end-start # save for VI timing later
                                    elbo_times[vi_iter].append(end-start + train_time)


                                    # Compute the predictive log-likelihood
                                    error = True
                                    while error:
                                        try:
                                            start = time.time()
                                            sample = model.sample_perm(k, data=data, inputs=covariates, reparam=False, device=device)
                                            pred_likelihood = model.predictive_ll(sample, N = 100, data_all=all_data, inputs_all=all_covariates)
                                            p_lls[vi_iter].append(pred_likelihood["obs"].item())
                                            end = time.time()

                                            p_ll_times[vi_iter].append(end-start + train_time)

                                            error = False
                                        except ValueError:
                                            pass

                                # Compute (an estimate of) the expectation for each variable in the model
                                start = time.time()
                                sample = model.sample_perm(k, data=data, inputs=covariates, reparam=False, device=device)
                                expectations[vi_iter].append(pp.mean(sample.weights()))
                                end = time.time()
                                expectation_times[vi_iter].append(end-start + train_time)

                                # Check that the results are valid
                                assert not t.any(t.tensor([t.any(t.isnan(tensor)) for _, tensor in expectations[vi_iter][i].items()])).item()
                                if useData:
                                    assert not np.isnan(p_lls[vi_iter][i])
                                    assert not np.isnan(elbos[vi_iter][i])

                                validSamples = True

                            except AssertionError:
                                # print("AssertionError")

                                # Remove results generated by this run
                                if useData:
                                    elbos[vi_iter] = elbos[vi_iter][:i]
                                    elbo_times[vi_iter] = elbo_times[vi_iter][:i]

                                    p_lls[vi_iter] = p_lls[vi_iter][:i]
                                    p_ll_times[vi_iter] = p_ll_times[vi_iter][:i]

                                expectations[vi_iter] = expectations[vi_iter][:i]
                                expectation_times[vi_iter] = expectation_times[vi_iter][:i]

                                pass
                        # input("Next run?")
                    if vi_iter % vi_iter_step != 0 or not useData:
                        start = time.time()
                        sample = model.sample_perm(k, data=data, inputs=covariates, reparam=True, device=device)
                        elbo = sample.elbo()
                        end = time.time()
                        elboTime = end-start

                    train_time_start = time.time()
                    opt.zero_grad()
                    # elbo = model.elbo(K=1)  # already computed elbo earlier
                    (-elbo).backward()
                    opt.step()
                    train_time_end = time.time()
                    train_time += elboTime + train_time_end - train_time_start

            # Compute variance/MSE of results, store w/ mean/std_err execution time
            for vi_iter in vi_iter_counts:

                rvs = list(expectations[vi_iter][0].keys())
                mean_vars = {rv: [] for rv in rvs}  # average element variance for each rv

                if useData:
                    expectation_means = {rv: sum([x[rv] for x in expectations[vi_iter]])/num_runs for rv in rvs}
                else:
                    expectation_means = {rv: sampledData[rv] for rv in rvs}  # use the true values for the sampled data

                sq_errs = {rv: [] for rv in rvs}

                for est in expectations[vi_iter]:
                    for rv in est:
                        sq_err = ((expectation_means[rv] - est[rv])**2).cpu()
                        sq_errs[rv].append(sq_err.rename(None))

                for rv in rvs:
                    mean_vars[rv] = float(t.mean(t.stack(sq_errs[rv])))

                for run in expectations[vi_iter]:
                    for rv in rvs:
                        run[rv].to("cpu")
                        del run[rv]

                expectations[vi_iter] = {}
                expectations[vi_iter]["time_mean"] = float(np.mean(expectation_times[vi_iter]))
                expectations[vi_iter]["time_std_err"] = float(np.std(expectation_times[vi_iter]))
                for rv in rvs:
                    expectations[vi_iter][rv] = {"mean_var": mean_vars[rv]}

            expectationsPerDataset.append(expectations.copy())

            # Clear up memory
            model.to("cpu")

            for x in [data, all_data, covariates, all_covariates]:
                for y in x.values():
                    y.to("cpu")
                    del y

            t.cuda.empty_cache()
            gc.collect()

        for vi_iter in vi_iter_counts:
            if useData:
                elbos[vi_iter] = {'mean': np.mean(elbos[vi_iter]),
                                    'std_err': np.std(elbos[vi_iter])/np.sqrt(num_runs),
                                    'time_mean': np.mean(elbo_times[vi_iter]),
                                    'time_std_err': np.std(elbo_times[vi_iter])/np.sqrt(num_runs)}


                p_lls[vi_iter] = {'mean': np.mean(p_lls[vi_iter]),
                                    'std_err': np.std(p_lls[vi_iter])/np.sqrt(num_runs),
                                    'time_mean': np.mean(p_ll_times[vi_iter]),
                                    'time_std_err': np.std(p_ll_times[vi_iter])/np.sqrt(num_runs)}

            # rvs = list(expectations[vi_iter][0].keys())
            expectations[vi_iter] = {}
            expectations[vi_iter]["time_mean"] = np.mean([expectationsPerDataset[d][vi_iter]["time_mean"] for d in range(num_datasets)])
            expectations[vi_iter]["time_std_err"] = np.mean([expectationsPerDataset[d][vi_iter]["time_std_err"] for d in range(num_datasets)])
            for rv in rvs:
                expectations[vi_iter][rv] = {"mean_var": np.mean([expectationsPerDataset[d][vi_iter][rv]["mean_var"] for d in range(num_datasets)])}


        # Write out results
        if useData:
            file = f'{resultsFolder}/vi_bus_breakdown_elbo_M{M}_J{J}_I{I}_lr{lr}.json'
            with open(file, 'w') as f:
                json.dump({f"vi_{lr}": elbos}, f, indent=4)

            file = f'{resultsFolder}/vi_bus_breakdown_p_ll_M{M}_J{J}_I{I}_lr{lr}.json'
            with open(file, 'w') as f:
                json.dump({f"vi_{lr}": p_lls}, f, indent=4)

            file = f'{resultsFolder}/vi_bus_breakdown_variance_M{M}_J{J}_I{I}_lr{lr}.json'
            with open(file, 'w') as f:
                json.dump({f"vi_{lr}": expectations}, f, indent=4)
        else:
            file = f'{resultsFolder}/vi_bus_breakdown_MSE_M{M}_J{J}_I{I}_lr{lr}.json'
            with open(file, 'w') as f:
                json.dump({f"vi_{lr}": expectations}, f, indent=4)

        print(f"Finished lr={lr}, useData={useData}")
