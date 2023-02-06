import torch as t
import torch.nn as nn
import alan
import time
import numpy as np
import json
from alan.experiment_utils import seed_torch
import alan.postproc as pp

device = t.device("cuda" if t.cuda.is_available() else "cpu")

Ns = [5,10]
Ms = [50,150,300]

for M in Ms:
    for N in Ns:
        t.cuda.empty_cache()

        sizes = {'plate_1':M, 'plate_2':N}
        d_z = 18
        seed_torch(0)
        def P(tr):
          '''
          Heirarchical Model
          '''

          tr.sample('mu_z', alan.Normal(t.zeros((d_z,)).to(device), t.ones((d_z,)).to(device)))
          tr.sample('psi_z', alan.Normal(t.zeros((d_z,)).to(device), t.ones((d_z,)).to(device)))

          tr.sample('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()), plates='plate_1')

          tr.sample('obs', alan.Bernoulli(logits = tr['z'] @ tr['x']))

        def Q(tr):
          '''
          Heirarchical Model
          '''

          tr.sample('mu_z', alan.Normal(t.zeros((d_z,)).to(device), t.ones((d_z,)).to(device)))
          tr.sample('psi_z', alan.Normal(t.zeros((d_z,)).to(device), t.ones((d_z,)).to(device)))

          tr.sample('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()), plates='plate_1')

          # tr.sample('obs', alan.Bernoulli(logits = tr['z'] @ tr['x']))

        # Load covariates
        covariates = {'x':t.load('data/weights_{0}_{1}.pt'.format(N,M)).to(device)}
        test_covariates = {'x':t.load('data/test_weights_{0}_{1}.pt'.format(N,M)).to(device)}
        all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
        covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)

        # Load data
        data = {'obs':t.load('data/data_y_{0}_{1}.pt'.format(N, M)).to(device)}
        test_data = {'obs':t.load('data/test_data_y_{0}_{1}.pt'.format(N, M)).to(device)}
        all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
        data['obs'] = data['obs'].rename('plate_1','plate_2')
        
        # Make the model
        model = alan.Model(P, Q, data, covariates)
        model.to(device)
        Ks = [1,3,10,30]#,50]

        # "MP" is defunct and in reality 'massively parallel' will come to mean "tmc_new" in the paper,
        # but we're generating results for this method anyway for now
        methods = ["MP", "tmc", "tmc_new", "global_k"]  
        
        elbos = {method: {k:[] for k in Ks} for method in methods}
        elbo_times = {method: {k:[] for k in Ks} for method in methods}

        p_lls = {method: {k:[] for k in Ks} for method in methods}
        p_ll_times = {method: {k:[] for k in Ks} for method in methods}

        mean_est = {method: {k:[] for k in Ks} for method in methods}
        mean_est_times = {method: {k:[] for k in Ks} for method in methods}

        for k in Ks:
            print(f"M={M}, N={N}, k={k}")

            num_runs = 1000
            for i in range(num_runs):
                # print("run", i)#, end=" ")

                # Compute the elbos
                start = time.time()
                elbos["MP"][k].append(model.elbo(k).item())#/num_runs)
                end = time.time()
                elbo_times["MP"][k].append(end-start)

                start = time.time()
                elbos["tmc_new"][k].append(model.elbo_tmc_new(k).item())#/num_runs)
                end = time.time()
                elbo_times["tmc_new"][k].append(end-start)

                start = time.time()
                elbos["tmc"][k].append(model.elbo_tmc(k).item())#/num_runs)
                end = time.time()
                elbo_times["tmc"][k].append(end-start)

                start = time.time()
                elbos["global_k"][k].append(model.elbo_global(k).item())#/num_runs)
                end = time.time()
                elbo_times["global_k"][k].append(end-start)


                # Compute the predictive log-likelihood

                # print(model.predictive_ll(1, 10, data_all=all_data, covariates_all=all_covariates, sample_method="MP")["obs"].item())
                for method in methods:
                    # print(method, end=". ")
                    error = True
                    while error:
                        try:
                            start = time.time()
                            p_lls[method][k].append(model.predictive_ll(k, 100, data_all=all_data, covariates_all=all_covariates, sample_method=method)["obs"].item())
                            end = time.time()
                            p_ll_times[method][k].append(end-start)
                            error = False
                        except ValueError:
                            # print("error")
                            pass

                # print()
                

                # Compute (an estimate of) the expectation for each variable in the model
                # Store the mean L1/L2/L_inf distance from each estimate to the sample mean
                start = time.time()
                mean_est["MP"][k].append(pp.mean(model.weights(k)))
                end=time.time()
                mean_est_times["MP"][k].append(end-start)

                start = time.time()
                mean_est["tmc"][k].append(pp.mean(model.weights_tmc(k)))
                end=time.time()
                mean_est_times["tmc"][k].append(end-start)

                start = time.time()
                mean_est["tmc_new"][k].append(pp.mean(model.weights_tmc_new(k)))
                end=time.time()
                mean_est_times["tmc_new"][k].append(end-start)

                start = time.time()
                mean_est["global_k"][k].append(pp.mean(model.weights_global(k)))
                end=time.time()
                mean_est_times["global_k"][k].append(end-start)


                # input("Next run?")

            # Compute mean/std_err of results, store w/ mean/std_err execution time 
            for method in methods:
                elbos[method][k] = {'mean': np.mean(elbos[method][k]),
                                    'std_err': np.std(elbos[method][k])/np.sqrt(num_runs),
                                    'time_mean': np.mean(elbo_times[method][k]),
                                    'time_std_err': np.std(elbo_times[method][k])/np.sqrt(num_runs)}

                
                p_lls[method][k] = {'mean': np.mean(p_lls[method][k]),
                                    'std_err': np.std(p_lls[method][k])/np.sqrt(num_runs),
                                    'time_mean': np.mean(p_ll_times[method][k]),
                                    'time_std_err': np.std(p_ll_times[method][k])/np.sqrt(num_runs)}

                 
                rvs = list(mean_est[method][k][0].keys())
                mean_est_means : dict = {rv: sum([x[rv] for x in mean_est[method][k]])/num_runs for rv in rvs}
               
                l1_norms = {rv: [] for rv in rvs}
                l2_norms = {rv: [] for rv in rvs}
                linf_norms = {rv: [] for rv in rvs}

                for est in mean_est[method][k]:
                    for rv in est:
                        err = mean_est_means[rv] - est[rv]
                        l1_norms[rv].append(np.linalg.norm(err, 1))
                        l2_norms[rv].append(np.linalg.norm(err, 2))
                        linf_norms[rv].append(np.linalg.norm(err, np.inf))

                mean_est[method][k] = {}
                mean_est[method][k]["time_mean"] = float(np.mean(mean_est_times[method][k]))
                mean_est[method][k]["time_std_err"] = float(np.std(mean_est_times[method][k])/np.sqrt(num_runs))
                for rv in rvs:
                    mean_est[method][k][rv] = {"l1_mean": float(np.mean(l1_norms[rv])),
                                               "l2_mean": float(np.mean(l2_norms[rv])), 
                                               "linf_mean": float(np.mean(linf_norms[rv])),
                                               "l1_std_err": float(np.std(l1_norms[rv])/np.sqrt(num_runs)),
                                               "l2_std_err": float(np.std(l2_norms[rv])/np.sqrt(num_runs)), 
                                               "linf_std_err": float(np.std(linf_norms[rv])/np.sqrt(num_runs))}


        file = 'results/movielens_elbo_N{0}_M{1}.json'.format(N,M)
        with open(file, 'w') as f:
            json.dump(elbos, f)

        file = 'results/movielens_p_ll_N{0}_M{1}.json'.format(N,M)
        with open(file, 'w') as f:
            json.dump(p_lls, f)

        file = 'results/movielens_mean_est_N{0}_M{1}.json'.format(N,M)
        with open(file, 'w') as f:
            json.dump(mean_est, f)
