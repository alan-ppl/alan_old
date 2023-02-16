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

device = "cpu" if forceCPU else t.device("cuda" if t.cuda.is_available() else "cpu")

Ns = [20]
Ms = [450]

for useData in [True, False]:
    for M in Ms:
        for N in Ns:
            print(device)
            sizes = {'plate_1':M, 'plate_2':N}
            d_z = 18
            seed_torch(0)
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

            # Load covariates
            covariates = {'x':t.load('data/weights_{0}_{1}.pt'.format(N,M)).to(device)}
            test_covariates = {'x':t.load('data/test_weights_{0}_{1}.pt'.format(N,M)).to(device)}
            all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
            covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)

            if useData:
                # Load data
                data = {'obs':t.load('data/data_y_{0}_{1}.pt'.format(N, M)).to(device)}
                test_data = {'obs':t.load('data/test_data_y_{0}_{1}.pt'.format(N, M)).to(device)}
                all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
                data['obs'] = data['obs'].rename('plate_1','plate_2')
            else:
                # Generate data
                sampledData = alan.sample(P, varnames=('obs','z','mu_z','psi_z'), platesizes=sizes, covariates=covariates)

                data = {'obs': sampledData['obs']}
                # N.B.: p_ll doesn't really makes sense for sampled data, but we could compute it anyway if we have test_data/all_data
                test_data = {'obs':t.load('data/test_data_y_{0}_{1}.pt'.format(N, M)).to(device)}
                all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
                data['obs'] = data['obs'].rename('plate_1','plate_2')

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

            for k in Ks["global_k"]:
                print(f"M={M}, N={N}, k={k}")

                for i in range(num_runs):
                    if verbose: 
                        if i % 250 == 0: print(f"{i+1}/{num_runs}")

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

                                        p_ll_times[method][k].append(end-start)

                                        error = False
                                    except ValueError:
                                        pass

                    # Compute (an estimate of) the expectation for each variable in the model
                    start = time.time()
                    expectations["global_k"][k].append(pp.mean(model.weights_global(k)))
                    end=time.time()
                    expectation_times["global_k"][k].append(end-start)

                    if k in Ks["tmc_new"]:
                        start = time.time()
                        expectations["tmc_new"][k].append(pp.mean(model.weights_tmc_new(k)))
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

                        expectations[method][k] = {}
                        expectations[method][k]["time_mean"] = float(np.mean(expectation_times[method][k]))
                        expectations[method][k]["time_std_err"] = float(np.std(expectation_times[method][k]))
                        for rv in rvs:
                            expectations[method][k][rv] = {"mean_var": mean_vars[rv]}


            if useData:
                file = f'{resultsFolder}/movielens_elbo_N{N}_M{M}.json'
                with open(file, 'w') as f:
                    json.dump(elbos, f)

                file = f'{resultsFolder}//movielens_p_ll_N{N}_M{M}.json'
                with open(file, 'w') as f:
                    json.dump(p_lls, f)

                file = f'{resultsFolder}/movielens_variance_N{N}_M{M}.json'
                with open(file, 'w') as f:
                    json.dump(expectations, f)
            else:
                file = f'{resultsFolder}/movielens_MSE_N{N}_M{M}.json'
                with open(file, 'w') as f:
                    json.dump(expectations, f)

            t.cuda.empty_cache()
            gc.collect()
