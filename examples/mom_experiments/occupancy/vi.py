import torch as t
import torch.nn as nn
import alan
import time
import numpy as np
import json
from alan.experiment_utils import seed_torch
import alan.postproc as pp
import gc
from occupancy import generate_model
import sys
import argparse

if t.cuda.is_available():
    t.cuda.synchronize()
script_start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cpu',           '-c',   type=bool,  nargs='?', default=False)
parser.add_argument('--verbose',       '-v',   type=bool,  nargs='?', default=False)
parser.add_argument('--num_runs',      '-n',   type=int,   nargs='?', default=1000,                  help="number of runs")
parser.add_argument('--dataset_seeds', '-d',   type=int,   nargs='+', default=[0],                   help="seeds for test/train split")
parser.add_argument('--results_tag',   '-t',   type=str,   nargs='?', default="",                    help="string to insert into results filenames")
parser.add_argument('--vi_iters',      '-i',   type=int,   nargs='?', default=100,                   help="number of VI iterations to perform")
parser.add_argument('--vi_lrs',        '-l',   type=float, nargs='+', default=[0.01, 0.001, 0.0001], help="learning rates for VI")
parser.add_argument('--k',             '-k',   type=int,   nargs='?', default=1,                     help="K for training the model")
parser.add_argument('--eval_ks',       '-e',   type=int,   nargs='+', default=[1,3,10,30],           help="Ks for evaluating the model")

arglist = sys.argv[1:]
args = parser.parse_args(arglist)
print(args)

forceCPU = args.cpu
verbose = args.verbose
num_runs = args.num_runs
dataset_seeds = args.dataset_seeds
results_tag = args.results_tag
num_vi_iters = args.vi_iters
lrs = args.vi_lrs
k = args.k
eval_ks = args.eval_ks
resultsFolder = "results"

device = "cpu" if forceCPU else t.device("cuda" if t.cuda.is_available() else "cpu")
print(device)

seed_torch(0)

M = 6
J = 12
I = 50
Returns = 5
sizes = {'plate_Years': M, 'plate_Birds':J, 'plate_Ids':I, 'plate_Replicate': 5}

for dataset_seed in dataset_seeds:
    print(f"Dataset seed: {dataset_seed}")
    # Run the experiment
    for lr in lrs:
        for useData in [False, True]:
            seed_torch(dataset_seed)

            P, Q, data, covariates, all_data, all_covariates, _ = generate_model(0,0, device, QModule=True, dataset_seed=dataset_seed)

            if not useData:
                # Generate data
                sampledData = alan.sample(P, platesizes=sizes, covariates=covariates)
                data = {'obs': sampledData['obs']}
                data['obs'] = data['obs']

            if useData:
                elbos = {e_k: {count :[] for count in range(num_vi_iters+1)} for e_k in eval_ks}
                elbo_times = {e_k: {count :[] for count in range(num_vi_iters+1)} for e_k in eval_ks}

                p_lls = {e_k: {count :[] for count in range(num_vi_iters+1)} for e_k in eval_ks}
                p_ll_times = {e_k: {count :[] for count in range(num_vi_iters+1)} for e_k in eval_ks}

            expectations = {e_k: {count :[] for count in range(num_vi_iters+1)} for e_k in eval_ks}
            expectation_times = {e_k: {count :[] for count in range(num_vi_iters+1)} for e_k in eval_ks}

            for i in range(num_runs):
                # Make the model
                model = alan.Model(P, Q(), data, covariates)
                model.to(device)

                opt = t.optim.Adam(model.parameters(), lr=lr)
                train_time = 0

                print('Hello')
                if verbose and i % 1 == 0: print(f"{i+1}/{num_runs}")

                for vi_iter in range(num_vi_iters+1):
                    if verbose and vi_iter % ((num_vi_iters // 4)) == 0: print(f"VI {vi_iter}/{num_vi_iters}")

                    validSamples = False
                    while not validSamples:
                        try:
                            # Compute the elbo for our training k
                            if t.cuda.is_available():
                                t.cuda.synchronize()
                            start = time.time()

                            obj = model.rws(k)
                            elbo = model.elbo(k, reparam=False)

                            if t.cuda.is_available():
                                t.cuda.synchronize()
                            end = time.time()

                            recent_elbo_time = end - start

                            if useData:
                                # Save elbo from training
                                elbos[k][vi_iter].append(elbo.item())
                                elbo_times[k][vi_iter].append(recent_elbo_time + train_time)

                            for e_k in eval_ks:
                                if useData:

                                    if e_k != k:
                                        # Compute the elbo for other evaluation ks
                                        if t.cuda.is_available():
                                            t.cuda.synchronize()
                                        start = time.time()

                                        elbo = model.elbo(e_k, reparam=False)

                                        if t.cuda.is_available():
                                            t.cuda.synchronize()
                                        end = time.time()

                                        elbos[e_k][vi_iter].append(elbo.item())
                                        elbo_times[e_k][vi_iter].append(end - start + train_time)

                                    # Compute the predictive log-likelihood
                                    error = True
                                    while error:
                                        try:
                                            if t.cuda.is_available():
                                                t.cuda.synchronize()
                                            start = time.time()

                                            p_lls[e_k][vi_iter].append(model.predictive_ll(e_k, 100, data_all=all_data, covariates_all=all_covariates)["obs"].item())

                                            if t.cuda.is_available():
                                                t.cuda.synchronize()
                                            end = time.time()

                                            p_ll_times[e_k][vi_iter].append(end-start + train_time)

                                            error = False
                                        except ValueError:
                                            print("NaN p_ll!")
                                            pass

                                # Compute (an estimate of) the expectation for each variable in the model
                                if t.cuda.is_available():
                                    t.cuda.synchronize()
                                start = time.time()

                                expectations[e_k][vi_iter].append(pp.mean(model.weights(e_k)))

                                if t.cuda.is_available():
                                    t.cuda.synchronize()
                                end = time.time()

                                expectation_times[e_k][vi_iter].append(end-start + train_time)

                                # Check that the results are valid
                                assert not t.any(t.tensor([t.any(t.isnan(tensor)) for _, tensor in expectations[e_k][vi_iter][i].items()])).item()
                                if useData:
                                    assert not np.isnan(p_lls[e_k][vi_iter][i])
                                    assert not np.isnan(elbos[e_k][vi_iter][i])

                                validSamples = True

                        except AssertionError:
                            print("NaN samples!")

                            # Remove results generated by this run
                            if useData:
                                elbos[e_k][vi_iter] = elbos[e_k][vi_iter][:i]
                                elbo_times[e_k][vi_iter] = elbo_times[e_k][vi_iter][:i]

                                p_lls[e_k][vi_iter] = p_lls[e_k][vi_iter][:i]
                                p_ll_times[e_k][vi_iter] = p_ll_times[e_k][vi_iter][:i]

                            expectations[e_k][vi_iter] = expectations[e_k][vi_iter][:i]
                            expectation_times[e_k][vi_iter] = expectation_times[e_k][vi_iter][:i]

                            pass

                    # Do a step of VI
                    if t.cuda.is_available():
                        t.cuda.synchronize()
                    train_time_start = time.time()

                    opt.zero_grad()
                    # elbo = model.elbo(K=1)
                    (-obj).backward()
                    opt.step()

                    if t.cuda.is_available():
                        t.cuda.synchronize()
                    train_time_end = time.time()

                    train_time += train_time_end - train_time_start + recent_elbo_time

            # Compute variance/MSE of results, store w/ mean/std_err execution time
            for e_k in eval_ks:
                for vi_iter in range(num_vi_iters+1):
                    if useData:
                        elbos[e_k][vi_iter] = {'mean': np.mean(elbos[e_k][vi_iter]),
                                               'std_err': np.std(elbos[e_k][vi_iter])/np.sqrt(num_runs),
                                               'time_mean': np.mean(elbo_times[e_k][vi_iter]),
                                               'time_std_err': np.std(elbo_times[e_k][vi_iter])/np.sqrt(num_runs)}


                        p_lls[e_k][vi_iter] = {'mean': np.mean(p_lls[e_k][vi_iter]),
                                               'std_err': np.std(p_lls[e_k][vi_iter])/np.sqrt(num_runs),
                                               'time_mean': np.mean(p_ll_times[e_k][vi_iter]),
                                               'time_std_err': np.std(p_ll_times[e_k][vi_iter])/np.sqrt(num_runs)}

                    rvs = list(expectations[e_k][vi_iter][0].keys())
                    mean_vars = {rv: [] for rv in rvs}  # average element variance for each rv

                    if useData:
                        expectation_means = {rv: sum([x[rv] for x in expectations[e_k][vi_iter]])/num_runs for rv in rvs}
                    else:
                        expectation_means = {rv: sampledData[rv] for rv in rvs}  # use the true values for the sampled data

                    sq_errs = {rv: [] for rv in rvs}

                    for est in expectations[e_k][vi_iter]:
                        for rv in est:
                            est[rv] = est[rv].align_as(expectation_means[rv])
                            sq_err = ((expectation_means[rv] - est[rv])**2).cpu()
                            sq_errs[rv].append(sq_err.rename(None))

                    for rv in rvs:
                        mean_vars[rv] = float(t.mean(t.stack(sq_errs[rv])))

                    for run in expectations[e_k][vi_iter]:
                        for rv in rvs:
                            run[rv].to("cpu")
                            del run[rv]

                    expectations[e_k][vi_iter] = {}
                    expectations[e_k][vi_iter]["time_mean"] = float(np.mean(expectation_times[e_k][vi_iter]))
                    expectations[e_k][vi_iter]["time_std_err"] = float(np.std(expectation_times[e_k][vi_iter]))
                    for rv in rvs:
                        expectations[e_k][vi_iter][rv] = {"mean_var": mean_vars[rv]}

            # Clean up memory
            model = model.to("cpu")

            for x in [data, all_data, covariates, all_covariates]:
                for y in x:
                    x[y] = x[y].to("cpu")
                del x

            t.cuda.empty_cache()
            gc.collect()

            if useData:
                file = f'{resultsFolder}/vi_occupancy{results_tag}_elbo_lr{lr}_{dataset_seed}.json'
                with open(file, 'w') as f:
                    json.dump({f"vi_{lr}": elbos}, f, indent=4)

                file = f'{resultsFolder}/vi_occupancy{results_tag}_p_ll_lr{lr}_{dataset_seed}.json'
                with open(file, 'w') as f:
                    json.dump({f"vi_{lr}": p_lls}, f, indent=4)

                file = f'{resultsFolder}/vi_occupancy{results_tag}_variance_lr{lr}_{dataset_seed}.json'
                with open(file, 'w') as f:
                    json.dump({f"vi_{lr}": expectations}, f, indent=4)
            else:
                file = f'{resultsFolder}/vi_occupancy{results_tag}_MSE_lr{lr}_{dataset_seed}.json'
                with open(file, 'w') as f:
                    json.dump({f"vi_{lr}": expectations}, f, indent=4)

            print(f"Finished lr={lr}, useData={useData}")

if t.cuda.is_available():
    t.cuda.synchronize()
script_end_time = time.time()
print(f"Finished. Took {script_end_time - script_start_time}s.")
