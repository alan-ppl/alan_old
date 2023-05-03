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

script_start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cpu',          '-c',   type=bool,  nargs='?', default=False)
parser.add_argument('--verbose',      '-v',   type=bool,  nargs='?', default=False)
parser.add_argument('--num_runs',     '-n',   type=int,   nargs='?', default=250,  help="number of runs")
parser.add_argument('--dataset_seed', '-d',   type=int,   nargs='?', default=0,    help="seed for test/train split")
parser.add_argument('--vi_iters',     '-i',   type=int,   nargs='?', default=100,  help="number of VI iterations to perform")
parser.add_argument('--vi_lr',        '-l',   type=float, nargs='?', default=0.1,  help="learning rate for VI")
parser.add_argument('--k',            '-k',   type=int,   nargs='?', default=1)

arglist = sys.argv[1:]
args = parser.parse_args(arglist)


forceCPU = args.cpu
verbose = args.verbose
num_runs = args.num_runs
dataset_seed = args.dataset_seed
num_vi_iters = args.vi_iters
lr = args.vi_lr
k = args.k

results_folder = "results"

device = "cpu" if forceCPU else t.device("cuda:0" if t.cuda.is_available() else "cpu")
print(device)

seed_torch(0)

M, N = 450, 20
sizes = {'plate_1':M, 'plate_2':N}
d_z = 18

# learningRates = [0.1, 0.01]#, 0.001, 0.0001, 0.00001]
# learningRates = [0.1]#,0.01,0.001,0.0001]

# Define the distributions
def P(tr, x):
    '''
    Heirarchical Model
    '''

    # tr('mu_z', alan.Normal(t.zeros((d_z,)).to(device),0.25*t.ones((d_z,)).to(device)))
    # tr('psi_z', alan.Normal(t.zeros((d_z,)).to(device), 0.25*t.ones((d_z,)).to(device)))

    tr('mu_z', alan.Normal(t.zeros((d_z,)).to(device),t.ones((d_z,)).to(device)))
    tr('psi_z', alan.Normal(t.zeros((d_z,)).to(device), t.ones((d_z,)).to(device)))

    tr('z', alan.Normal(tr['mu_z'], tr['psi_z'].exp()), plates='plate_1')

    if tr['z'].dtype == t.float32:
        tr('obs', alan.Bernoulli(logits = tr['z'] @ x.float()))
    else:
        tr('obs', alan.Bernoulli(logits = tr['z'] @ x))

class Q(alan.AlanModule):
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


    def forward(self, tr, x):
        # tr('mu_z', alan.Normal(self.m_mu_z, 0.25*self.log_theta_mu_z.exp()))#, multi_sample=False if local else True)
        # tr('psi_z', alan.Normal(self.m_psi_z, 0.25*self.log_theta_psi_z.exp()))#, multi_sample=False if local else True)

        tr('mu_z', alan.Normal(self.m_mu_z, self.log_theta_mu_z.exp()))#, multi_sample=False if local else True)
        tr('psi_z', alan.Normal(self.m_psi_z, self.log_theta_psi_z.exp()))#, multi_sample=False if local else True)

        tr('z', alan.Normal(self.mu, self.log_sigma.exp()))


# Run the experiment
for useData in [False, True]:

    # Load covariates
    covariates = {'x':t.load(f'data/weights_{N}_{M}_{dataset_seed}.pt').to(device)}
    test_covariates = {'x':t.load(f'data/test_weights_{N}_{M}_{dataset_seed}.pt').to(device)}
    all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
    covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)


    if useData:
        elbos = {count :[] for count in range(num_vi_iters+1)}
        elbo_times = {count:[] for count in range(num_vi_iters+1)}

        p_lls = {count:[] for count in range(num_vi_iters+1)}
        p_ll_times = {count:[] for count in range(num_vi_iters+1)}

    expectations = {count:[] for count in range(num_vi_iters+1)} 
    expectation_times = {count:[] for count in range(num_vi_iters+1)}

    for i in range(num_runs):
        # Make the model
        model = alan.Model(P, Q())#, data, covariates)

        model.to(device)

        if useData:
            # Load data
            data = {'obs':t.load(f'data/data_y_{N}_{M}_{dataset_seed}.pt').to(device)}
            test_data = {'obs':t.load(f'data/test_data_y_{N}_{M}_{dataset_seed}.pt').to(device)}
            all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
            data['obs'] = data['obs'].rename('plate_1','plate_2')
        else:
            # Generate data
            # sampledData = alan.sample(P, varnames=('obs','z','mu_z','psi_z'), platesizes=sizes, covariates=covariates)

            # data = {'obs': sampledData['obs']}
            # test_data = {'obs':t.load(f'data/test_data_y_{N}_{M}_{dataset_seed}.pt').to(device)}
            # all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}

            sampledData = model.sample_prior(platesizes = sizes, inputs = covariates, device=device)
            data = {'obs': sampledData['obs']}
            data['obs'] = data['obs'].rename('plate_1','plate_2')

        opt = t.optim.Adam(model.parameters(), lr=lr)

        train_time = 0

        if verbose: 
            if i % 100 == 0: 
                # print(f"{i+1}/{num_runs}")
                print(f"run {i}")

        for vi_iter in range(num_vi_iters+1):
            if verbose and vi_iter % 250 == 0: print(f"{vi_iter}/{num_vi_iters}")

            # Compute the elbo
            start = time.time()
            sample = model.sample_perm(k, data=data, inputs=covariates, reparam=True, device=device)
            elbo = sample.elbo()
            end   = time.time()
            elbo_time = end-start

            if useData:
                elbos[vi_iter].append(elbo.item())
                elbo_times[vi_iter].append(elbo_time + train_time)

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

            # input("Next run?")
            
            train_time_start = time.time()
            opt.zero_grad()
            # elbo = model.elbo(K=1)
            (-elbo).backward()
            opt.step()
            train_time_end = time.time()
            train_time += train_time_end - train_time_start

    # Compute variance/MSE of results, store w/ mean/std_err execution time 
    for vi_iter in range(num_vi_iters+1):
        if useData:
            elbos[vi_iter] = {'mean': np.mean(elbos[vi_iter]),
                                'std_err': np.std(elbos[vi_iter])/np.sqrt(num_runs),
                                'time_mean': np.mean(elbo_times[vi_iter]),
                                'time_std_err': np.std(elbo_times[vi_iter])/np.sqrt(num_runs)}

        
            p_lls[vi_iter] = {'mean': np.mean(p_lls[vi_iter]),
                                'std_err': np.std(p_lls[vi_iter])/np.sqrt(num_runs),
                                'time_mean': np.mean(p_ll_times[vi_iter]),
                                'time_std_err': np.std(p_ll_times[vi_iter])/np.sqrt(num_runs)}
    
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

        # Clean up memory
        model.to("cpu")
        tensors_to_remove = [data,  covariates]
        if useData:
            tensors_to_remove += [all_data, test_data, all_covariates, test_covariates]
        for x in tensors_to_remove:
            for y in x.values():
                y.to("cpu")
                del y

        t.cuda.empty_cache()
        gc.collect()

      
    if useData:
        file = f'{results_folder}/vi_movielens_elbo_lr{lr}.json'
        with open(file, 'w') as f:
            json.dump({f"vi_{lr}": elbos}, f, indent=4)

        file = f'{results_folder}/vi_movielens_p_ll_lr{lr}.json'
        with open(file, 'w') as f:
            json.dump({f"vi_{lr}": p_lls}, f, indent=4)

        file = f'{results_folder}/vi_movielens_variance_lr{lr}.json'
        with open(file, 'w') as f:
            json.dump({f"vi_{lr}": expectations}, f, indent=4)
    else:
        file = f'{results_folder}/vi_movielens_MSE_lr{lr}.json'
        with open(file, 'w') as f:
            json.dump({f"vi_{lr}": expectations}, f, indent=4)

    print(f"Finished lr={lr}, useData={useData}")

print(f"Took {time.time() - script_start_time}s.")