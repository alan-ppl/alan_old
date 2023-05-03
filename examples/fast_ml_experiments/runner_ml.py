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
import pickle

from alan.experiment_utils import seed_torch, n_mean


seed_torch(0)
### Maybe check if data is empty and run data making script before experiment?


print('...', flush=True)

@hydra.main(version_base=None, config_path='config', config_name='conf')
def run_experiment(cfg):
    print('ML')
    print(cfg)
    # writer = SummaryWriter(log_dir='runs/' + cfg.dataset + '/' + cfg.model + '/')
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


    Ks = cfg.training.Ks

    M = cfg.training.M
    N = cfg.training.N

    spec = importlib.util.spec_from_file_location(cfg.model, cfg.dataset + '/' + cfg.model + '.py')
    foo = importlib.util.module_from_spec(spec)
    sys.modules[cfg.model] = foo
    spec.loader.exec_module(foo)


    for K in Ks:
        print(K)
        per_seed_obj = np.zeros((cfg.training.num_runs,cfg.training.num_iters), dtype=np.float32)
        pred_liks = np.zeros((cfg.training.num_runs,cfg.training.num_iters), dtype=np.float32)
        sq_errs = np.zeros((cfg.training.num_runs,cfg.training.num_iters), dtype=np.float32)
        times = np.zeros((cfg.training.num_runs,cfg.training.num_iters), dtype=np.float32)
        nans = np.asarray([0]*cfg.training.num_runs)
        final_pred_lik = np.zeros((cfg.training.num_runs,), dtype=np.float32)
        final_pred_lik_for_K = np.zeros((cfg.training.num_runs,len(Ks)), dtype=np.float32)
        for i in range(cfg.training.num_runs):
            seed_torch(i)
            P, Q, data, covariates, test_data, test_covariates, all_data, all_covariates, sizes = foo.generate_model(N,M, device, cfg.training.ML, i, cfg.use_data)

            if not cfg.use_data:
                data_prior = data
                data = {'obs':data.pop('obs')}
                test_data = {'obs':test_data.pop('obs')}

            per_seed_elbos = []



            model = alan.Model(P, Q())
            model.to(device)


            for j in range(cfg.training.num_iters):
                start = time.time()
                sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=device)
                elbo = sample.elbo().item()
                per_seed_obj[i,j] = (elbo)
                model.update(cfg.training.lr, sample)

                times[i,j] = (time.time() - start)

                #Predictive Log Likelihoods
                if cfg.training.pred_ll.do_pred_ll:
                    success=False
                    for k in range(10):
                        try:
                            sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=device)
                            pred_likelihood = model.predictive_ll(sample, N = cfg.training.pred_ll.num_pred_ll_samples, data_all=all_data, inputs_all=all_covariates)
                            pred_liks[i,j] = pred_likelihood['obs'].item()
                            success=True
                        except:
                            print('nan pred likelihood!')
                            nans[i] += 1
                        if success:
                            break
                    if not success:
                        pred_liks[i,j] = np.nan


                if cfg.do_moments:
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
                        sq_errs[i,j] += ((expectation_means[rv].cpu() - exps[rv].cpu())**2).rename(None).sum().cpu()/(len(rvs))


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

            for K_run in range(len(Ks)):
                for k in range(10):
                    try:
                        sample = model.sample_perm(Ks[K_run], data=data, inputs=covariates, reparam=False, device=device)
                        pred_likelihood = model.predictive_ll(sample, N = cfg.training.pred_ll.num_pred_ll_samples, data_all=all_data, inputs_all=all_covariates)
                        final_pred_lik_for_K[i, K_run] = pred_likelihood['obs'].item()
                        # print(pred_liks[i,j])
                        break
                    except:
                        final_pred_lik_for_K[i, K_run] = np.nan
                        print('nan pred likelihood!')
            ###
            # SAVING MODELS DOESN'T WORK YET
            ###
            if not os.path.exists(cfg.dataset + '/' + 'results/' + cfg.model + '/'):
                os.makedirs(cfg.dataset + '/' + 'results/' + cfg.model + '/')

            # t.save(model.state_dict(), cfg.dataset + '/' + 'results/' + '{0}_{1}'.format(cfg.model, i))
        results_dict = {'objs':per_seed_obj,
                                 'pred_likelihood':pred_liks,
                                 'times':times,
                                 'nans':(nans/cfg.training.num_runs).tolist(),
                                 'sq_errs':sq_errs,
                                 'final_pred_lik_K=30':final_pred_lik,
                                 'final_pred_lik_for_K':final_pred_lik_for_K}


        file = cfg.dataset + '/results/' + cfg.model + '/ML_{}'.format(cfg.training.num_iters) + '_{}_'.format(cfg.training.lr) + 'K{0}_{1}.pkl'.format(K,cfg.use_data)
        with open(file, 'wb') as f:
            pickle.dump(results_dict, f)


if __name__ == "__main__":
    run_experiment()
