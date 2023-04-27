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

num_datasets = 3

num_vi_iters = 100
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

device = "cpu" if forceCPU else t.device("cuda" if t.cuda.is_available() else "cpu")
print(device)

seed_torch(0)

M, N = 450, 20
sizes = {'plate_1':M, 'plate_2':N}
d_z = 18

k = 1
learningRates = [0.1, 0.01]#, 0.001, 0.0001, 0.00001]
learningRates = [0.1]#,0.01,0.001,0.0001]

# Define the distributions
def P(tr):
    '''
    Heirarchical Model
    '''

    tr.sample('mu_z', alan.Normal(t.zeros((d_z,)).to(device),0.25*t.ones((d_z,)).to(device)))
    tr.sample('psi_z', alan.Normal(t.zeros((d_z,)).to(device), 0.25*t.ones((d_z,)).to(device)))

    tr.sample('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()), plates='plate_1')

    tr.sample('obs', alan.Bernoulli(logits = tr['z'] @ tr['x']))

class Q(alan.QModule):
    def __init__(self):
        super().__init__()
        #mu_z
        self.m_mu_z = nn.Parameter(t.zeros((d_z,)))
        self.log_theta_mu_z = nn.Parameter(t.zeros((d_z,)))
        #psi_z
        self.m_psi_z = nn.Parameter(t.zeros((d_z,)))
        self.log_theta_psi_z = nn.Parameter(t.zeros((d_z,)))

        #z
        self.mu = nn.Parameter(t.zeros((M,d_z), names=('plate_1',None)))
        self.log_sigma = nn.Parameter(t.zeros((M,d_z), names=('plate_1',None)))


    def forward(self, tr):
        tr.sample('mu_z', alan.Normal(self.m_mu_z, 0.25*self.log_theta_mu_z.exp()))#, multi_sample=False if local else True)
        tr.sample('psi_z', alan.Normal(self.m_psi_z, 0.25*self.log_theta_psi_z.exp()))#, multi_sample=False if local else True)

        tr.sample('z', alan.Normal(self.mu, self.log_sigma.exp()))


# Run the experiment
for lr in learningRates:
    for useData in [False, True]:
        expectationsPerDataset = []

        for datasetSeed in range(num_datasets):
            # Load covariates
            covariates = {'x':t.load(f'data/weights_{N}_{M}_{datasetSeed}.pt').to(device)}
            test_covariates = {'x':t.load(f'data/test_weights_{N}_{M}_{datasetSeed}.pt').to(device)}
            all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
            covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)

            if useData:
                # Load data
                data = {'obs':t.load(f'data/data_y_{N}_{M}_{datasetSeed}.pt').to(device)}
                test_data = {'obs':t.load(f'data/test_data_y_{N}_{M}_{datasetSeed}.pt').to(device)}
                all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
                data['obs'] = data['obs'].rename('plate_1','plate_2')
            else:
                # Generate data
                sampledData = alan.sample(P, varnames=('obs','z','mu_z','psi_z'), platesizes=sizes, covariates=covariates)

                data = {'obs': sampledData['obs']}
                test_data = {'obs':t.load(f'data/test_data_y_{N}_{M}_{datasetSeed}.pt').to(device)}
                all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
                data['obs'] = data['obs'].rename('plate_1','plate_2')

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
                model = alan.Model(P, Q(), data, covariates)

                model.to(device)

                opt = t.optim.Adam(model.parameters(), lr=lr)

                train_time = 0

                if verbose: 
                    if i % 100 == 0: print(f"{i+1}/{num_runs}")

                for vi_iter in range(num_vi_iters+1):

                    if vi_iter % vi_iter_step == 0:
                        # if verbose: print(f"{vi_iter}/{num_vi_iters}")

                        if useData:
                            # Compute the elbo
                            start = time.time()
                            elbos[vi_iter].append(model.elbo(k).item())
                            end = time.time()
                            elbo_times[vi_iter].append(end-start + train_time)


                            # Compute the predictive log-likelihood
                            error = True
                            while error:
                                try:
                                    start = time.time()
                                    p_lls[vi_iter].append(model.predictive_ll(k, 100, data_all=all_data, covariates_all=all_covariates)["obs"].item())
                                    end = time.time()

                                    p_ll_times[vi_iter].append(end-start + train_time)

                                    error = False
                                except ValueError:
                                    pass

                        # Compute (an estimate of) the expectation for each variable in the model
                        start = time.time()
                        expectations[vi_iter].append(pp.mean(model.weights(k)))
                        end = time.time()
                        expectation_times[vi_iter].append(end-start + train_time)

                        # input("Next run?")
                    
                    train_time_start = time.time()
                    opt.zero_grad()
                    elbo = model.elbo(K=1)
                    (-elbo).backward()
                    opt.step()
                    train_time_end = time.time()
                    train_time += train_time_end - train_time_start

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

            # Clean up memory
            model.to("cpu")

            for x in [data, test_data, all_data, covariates, test_covariates, all_covariates]:
                for y in x.values():
                    y.to("cpu")
                    del y

            t.cuda.empty_cache()
            gc.collect()

        # Average out over datasets and write out results
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
                    
        if useData:
            file = f'{resultsFolder}/vi_movielens_elbo_N{N}_M{M}_lr{lr}.json'
            with open(file, 'w') as f:
                json.dump({f"vi_{lr}": elbos}, f, indent=4)

            file = f'{resultsFolder}/vi_movielens_p_ll_N{N}_M{M}_lr{lr}.json'
            with open(file, 'w') as f:
                json.dump({f"vi_{lr}": p_lls}, f, indent=4)

            file = f'{resultsFolder}/vi_movielens_variance_N{N}_M{M}_lr{lr}.json'
            with open(file, 'w') as f:
                json.dump({f"vi_{lr}": expectations}, f, indent=4)
        else:
            file = f'{resultsFolder}/vi_movielens_MSE_N{N}_M{M}_lr{lr}.json'
            with open(file, 'w') as f:
                json.dump({f"vi_{lr}": expectations}, f, indent=4)

        print(f"Finished lr={lr}, useData={useData}")

