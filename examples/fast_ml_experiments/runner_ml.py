import torch as t
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import alan

import os
import numpy as np
import itertools
import time
import random
import hydra
import importlib.util
import sys
import json

from alan.experiment_utils import seed_torch


seed_torch(0)
### Maybe check if data is empty and run data making script before experiment?


print('...', flush=True)

@hydra.main(version_base=None, config_path='config', config_name='conf')
def run_experiment(cfg):
    print(cfg)
    # writer = SummaryWriter(log_dir='runs/' + cfg.dataset + '/' + cfg.model + '/')
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

    results_dict = {}

    Ks = cfg.training.Ks

    M = cfg.training.M
    N = cfg.training.N

    spec = importlib.util.spec_from_file_location(cfg.model, cfg.dataset + '/' + cfg.model + '.py')
    foo = importlib.util.module_from_spec(spec)
    sys.modules[cfg.model] = foo
    spec.loader.exec_module(foo)


    for K in Ks:
        print(K,M,N)
        results_dict[N] = results_dict.get(N, {})
        results_dict[N][M] = results_dict[N].get(M, {})
        results_dict[N][M][K] = results_dict[N][M].get(K, {})
        per_seed_obj = np.zeros((cfg.training.num_runs,cfg.training.num_iters))
        pred_liks = np.zeros((cfg.training.num_runs,cfg.training.num_iters))
        times = np.zeros((cfg.training.num_runs,cfg.training.num_iters))
        for i in range(cfg.training.num_runs):
            P, Q, data, covariates, test_data, test_covariates, all_data, all_covariates = foo.generate_model(N,M, device, cfg.training.ML, i)

            per_seed_elbos = []
            start = time.time()
            seed_torch(i)

            model = alan.Model(P, Q())
            model.to(device)

            nans = 0

            for j in range(cfg.training.num_iters):
                sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=device)
                elbo = sample.elbo().item()
                per_seed_obj[i,j] = (elbo)
                model.update(lr, sample)

                times[i,j] = ((time.time() - start)/cfg.training.num_iters)

                if j % 100 == 0:
                    print("Iteration: {0}, ELBO: {1:.2f}".format(j,elbo))

                if cfg.training.pred_ll.do_pred_ll:
                    success=False
                    for k in range(10):
                        try:
                            sample = model.sample_perm(K, data=test_data, inputs=test_covariates, reparam=False, device=device)
                            pred_likelihood = model.predictive_ll(sample, N = cfg.training.pred_ll.num_pred_ll_samples, data_all=all_data, inputs_all=all_covariates)
                            pred_liks[i,j] = pred_likelihood['obs'].item()
                            success=True
                        except:
                            print('nan pred likelihood!')
                            nans += 1

                    if not success:
                        pred_liks[i,j] = np.nan
                else:
                    pred_liks[i,j] = (0)

            ###
            # SAVING MODELS DOESN'T WORK YET
            ###
            if not os.path.exists(cfg.dataset + '/' + 'results/' + cfg.model + '/'):
                os.makedirs(cfg.dataset + '/' + 'results/' + cfg.model + '/')
            #
            # t.save(model.state_dict(), cfg.dataset + '/' + 'results/' + '{0}_{1}'.format(cfg.model, i))

        results_dict[N][M][K] = {'objs':np.nanmean(per_seed_obj, axis=0, keepdims=False).tolist(), 'obj_stds':np.nanstd(per_seed_obj, axis=0, keepdims=False).tolist(), 'pred_likelihood':np.nanmean(pred_liks, axis=0, keepdims=False).tolist(), 'pred_likelihood_std':np.nanstd(pred_liks, axis=0, keepdims=False).tolist(), 'avg_time':np.nanmean(times, axis=0, keepdims=False).tolist(), 'time':np.cumsum(np.nanmean(times, axis=0, keepdims=False), axis=-1).tolist(), 'nans':nans/num_runs}

        file = cfg.dataset + '/results/' + cfg.model + '/ML_{}'.format(cfg.training.ML) + '_{}'.format(cfg.training.num_iters) + '_{}_'.format(cfg.training.lr) + 'N{0}_M{1}_K{2}.json'.format(N,M,K)
        with open(file, 'w') as f:
            json.dump(results_dict, f)

if __name__ == "__main__":
    run_experiment()
