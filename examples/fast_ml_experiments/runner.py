import torch as t
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import alan
import alan.postproc as pp

import os
import numpy as np
import itertools
import time
import random
import hydra
import importlib.util
import sys
import json

from alan.experiment_utils import seed_torch, n_mean


seed_torch(0)
### Maybe check if data is empty and run data making script before experiment?


print('...', flush=True)


@hydra.main(version_base=None, config_path='config', config_name='conf')
def run_experiment(cfg):
    print('VI')
    print(cfg)
    # writer = SummaryWriter(log_dir='runs/' + cfg.dataset + '/' + cfg.model + '/')
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

    results_dict = {}

    Ks = cfg.training.Ks

    M = cfg.training.M
    N = cfg.training.N

    spec = importlib.util.spec_from_file_location(cfg.model, cfg.dataset + '/' + cfg.model + '_VI.py')
    foo = importlib.util.module_from_spec(spec)
    sys.modules[cfg.model] = foo
    spec.loader.exec_module(foo)



    for K in Ks:
        print(K,M,N)
        results_dict[N] = results_dict.get(N, {})
        results_dict[N][M] = results_dict[N].get(M, {})
        results_dict[N][M][K] = results_dict[N][M].get(K, {})
        per_seed_obj = np.zeros((cfg.training.num_runs,cfg.training.num_iters), dtype=np.float32)
        pred_liks = np.zeros((cfg.training.num_runs,cfg.training.num_iters), dtype=np.float32)
        sq_errs = np.zeros((cfg.training.num_runs,cfg.training.num_iters), dtype=np.float32)
        times = np.zeros((cfg.training.num_runs,cfg.training.num_iters), dtype=np.float32)
        nans = np.asarray([0]*cfg.training.num_runs)
        final_pred_lik = np.zeros((cfg.training.num_runs,), dtype=np.float32)
        for i in range(cfg.training.num_runs):
            P, Q, data, covariates, test_data, test_covariates, all_data, all_covariates, sizes = foo.generate_model(N,M, device, cfg.training.ML, i)


            seed_torch(i)

            model = alan.Model(P, Q())
            model.to(device)

            if not cfg.use_data:
                data_prior = model.sample_prior(platesizes = sizes, inputs = covariates, device=device)
                data = {'obs': data_prior['obs']}

            opt = t.optim.Adam(model.parameters(), lr=cfg.training.lr)

            for j in range(cfg.training.num_iters):
                start = time.time()
                opt.zero_grad()
                sample = model.sample_perm(K, data=data, inputs=covariates, reparam=True, device=device)
                elbo = sample.elbo()
                per_seed_obj[i,j] = elbo.item()
                (-elbo).backward()
                opt.step()
                times[i,j] = (time.time() - start)


                #Predictive Log Likelihoods
                if cfg.training.pred_ll.do_pred_ll and cfg.use_data:
                    success=False
                    for k in range(10):
                        try:
                            sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=device)
                            pred_likelihood = model.predictive_ll(sample, N = cfg.training.pred_ll.num_pred_ll_samples, data_all=all_data, inputs_all=all_covariates)
                            pred_liks[i,j] = pred_likelihood['obs'].item()
                            success=True
                            # print(pred_liks[i,j])
                        except:
                            print('nan pred likelihood!')
                            nans[i] += 1
                        if success:
                            break
                    if not success:
                        pred_liks[i,j] = np.nan


                #MSE/Variance of first moment
                sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=device)
                exps = pp.mean(sample.weights())

                rvs = list(exps.keys())
                if cfg.use_data:
                    expectation_means = {rv: exps[rv]/cfg.training.num_runs for rv in rvs}
                else:
                    expectation_means = {rv: data_prior[rv] for rv in rvs}  # use the true values for the sampled data

                sq_err = 0
                for rv in rvs:
                    sq_errs[i,j] += ((expectation_means[rv] - exps[rv])**2).rename(None).sum().cpu()/(len(rvs))


                if j % 100 == 0:
                    print("Iteration: {0}, ELBO: {1:.2f}".format(j,elbo))
                    print("Iteration: {0}, Predll: {1:.2f}".format(j,pred_liks[i,j]))

            for k in range(10):
                try:
                    sample = model.sample_perm(30, data=data, inputs=covariates, reparam=False, device=device)
                    pred_likelihood = model.predictive_ll(sample, N = cfg.training.pred_ll.num_pred_ll_samples, data_all=all_data, inputs_all=all_covariates)
                    final_pred_lik[i] = pred_likelihood['obs'].item()
                    # print(pred_liks[i,j])
                    break
                except:
                    final_pred_lik[i] = np.nan
                    print('nan pred likelihood!')

            ###
            # SAVING MODELS DOESN'T WORK YET
            ###
            if not os.path.exists(cfg.dataset + '/' + 'results/' + cfg.model + '/'):
                os.makedirs(cfg.dataset + '/' + 'results/' + cfg.model + '/')

            # t.save(model.state_dict(), cfg.dataset + '/' + 'results/' + '{0}_{1}'.format(cfg.model, i))
        if cfg.plotting.average:
            per_seed_obj = n_mean(per_seed_obj, cfg.plotting.n_avg)
            pred_liks = n_mean(pred_liks, cfg.plotting.n_avg)
            sq_errs = n_mean(sq_errs, cfg.plotting.n_avg)
            results_dict[N][M][K] = {'objs':np.nanmean(per_seed_obj, axis=0, keepdims=False).tolist(),
                                     'obj_stds':(np.nanstd(per_seed_obj, axis=0, keepdims=False) / np.sqrt(cfg.training.num_runs)).tolist(),
                                     'pred_likelihood':np.nanmean(pred_liks, axis=0, keepdims=False).tolist(),
                                     'pred_likelihood_std':(np.nanstd(pred_liks, axis=0, keepdims=False) / np.sqrt(cfg.training.num_runs)).tolist(),
                                     'avg_time':np.nanmean(times, axis=0, keepdims=False).tolist()[::cfg.plotting.n_avg],
                                     'time':np.cumsum(np.nanmean(times, axis=0, keepdims=False), axis=-1).tolist()[::cfg.plotting.n_avg],
                                     'nans':(nans/cfg.training.num_runs).tolist(),
                                     'sq_errs':np.nanmean(sq_errs, axis=0, keepdims=False).tolist(),
                                     'sq_errs_std':(np.nanstd(sq_errs, axis=0, keepdims=False) / np.sqrt(cfg.training.num_runs)).tolist(),
                                     'final_pred_lik_K=30':np.nanmean(final_pred_lik, axis=0, keepdims=False).tolist(),
                                     'final_pred_lik_K=30_stderr':(np.nanstd(final_pred_lik, axis=0, keepdims=False)/ np.sqrt(cfg.training.num_runs)).tolist(),}

        else:
            results_dict[N][M][K] = {'objs':np.nanmean(per_seed_obj, axis=0, keepdims=False).tolist(),
                                     'obj_stds':(np.nanstd(per_seed_obj, axis=0, keepdims=False) / np.sqrt(cfg.training.num_runs)).tolist(),
                                     'pred_likelihood':np.nanmean(pred_liks, axis=0, keepdims=False).tolist(),
                                     'pred_likelihood_std':(np.nanstd(pred_liks, axis=0, keepdims=False) / np.sqrt(cfg.training.num_runs)).tolist(),
                                     'avg_time':np.nanmean(times, axis=0, keepdims=False).tolist(),
                                     'time':np.cumsum(np.nanmean(times, axis=0, keepdims=False), axis=-1).tolist(),
                                     'nans':(nans/cfg.training.num_runs).tolist(),
                                     'sq_errs':np.nanmean(sq_errs, axis=0, keepdims=False).tolist(),
                                     'sq_errs_std':(np.nanstd(sq_errs, axis=0, keepdims=False) / np.sqrt(cfg.training.num_runs)).tolist(),
                                     'final_pred_lik_K=30':np.nanmean(final_pred_lik, axis=0, keepdims=False).tolist(),
                                     'final_pred_lik_K=30_stderr':(np.nanstd(final_pred_lik, axis=0, keepdims=False)/ np.sqrt(cfg.training.num_runs)).tolist(),}

        file = cfg.dataset + '/results/' + cfg.model + '/VI_{}'.format(cfg.training.num_iters) + '_{}_'.format(cfg.training.lr) + 'N{0}_M{1}_K{2}_{3}.json'.format(N,M,K,cfg.use_data)
        with open(file, 'w') as f:
            json.dump(results_dict, f)

if __name__ == "__main__":
    run_experiment()
