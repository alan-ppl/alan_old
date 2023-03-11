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

# parser = argparse.ArgumentParser(description='Run an experiment.')
#
# parser.add_argument('f', type=str,
#                     help='Yaml file describing experiment')
# args = parser.parse_args()
# f = args.f
#
# config_path = f.rsplit('/', 1)[0]
# config_name = f.rsplit('/', 1)[1]
#
# print(config_path)
# print(config_name)
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
    # foo.MyClass()
    # experiment = importlib.import_module(cfg.model.model, 'Deeper_Hier_Regression')

    P, Q, data, covariates, test_data, test_covariates, all_data, all_covariates = foo.generate_model(N,M, device, cfg.training.ML)
    for K in Ks:
        print(K,M,N)
        results_dict[N] = results_dict.get(N, {})
        results_dict[N][M] = results_dict[N].get(M, {})
        results_dict[N][M][K] = results_dict[N][M].get(K, {})
        per_seed_obj = []
        objs = []
        pred_liks = []
        times = []
        for i in range(cfg.training.num_runs):
            per_seed_elbos = []
            start = time.time()
            seed_torch(i)

            model = alan.Model(P, Q())#.condition(data=data)
            model.to(device)


            for j in range(cfg.training.num_iters):
                sample = model.sample_same(K, data=data, inputs=covariates, reparam=False, device=device)
                elbo = sample.elbo().item()
                per_seed_obj.append(elbo)
                model.update(cfg.training.lr, sample)

                # writer.add_scalar('Objective/Run number {}/{}'.format(i, K), elbo, j)
                if j % 10 == 0:
                    print("Iteration: {0}, ELBO: {1:.2f}".format(j,elbo))



            if cfg.training.pred_ll.do_pred_ll:
                sample = model.sample_same(K, data=test_data, inputs=test_covariates, reparam=False, device=device)
                pred_likelihood = model.predictive_ll(sample, N = cfg.training.pred_ll.num_pred_ll_samples, data_all=all_data, inputs_all=all_covariates)
                pred_liks.append(pred_likelihood['obs'].item())
            else:
                pred_liks.append(0)
            objs.append(np.mean(per_seed_obj[-50:]))
            times.append((time.time() - start)/cfg.training.num_iters)
            # writer.add_scalar('Time/Run number {}'.format(i,K), times[-1], K)
            # writer.add_scalar('Predictive Log Likelihood/Run number {}'.format(i,K), pred_liks[-1], K)

            ###
            # SAVING MODELS DOESN'T WORK YET
            ###
            if not os.path.exists(cfg.dataset + '/' + 'results/' + cfg.model + '/'):
                os.makedirs(cfg.dataset + '/' + 'results/' + cfg.model + '/')
            #
            # t.save(model.state_dict(), cfg.dataset + '/' + 'results/' + '{0}_{1}'.format(cfg.model, i))

        results_dict[N][M][K] = {'final_obj':np.mean(objs),'final_obj_std':np.std(objs), 'pred_likelihood':np.mean(pred_liks), 'pred_likelihood_std':np.std(pred_liks), 'objs': objs, 'pred_liks':pred_liks, 'avg_time':np.mean(times), 'std_time':np.std(times)}

    file = cfg.dataset + '/results/' + cfg.model + '/ML_{}'.format(cfg.training.ML) + '_{}_'.format(cfg.training.lr) + '_' + 'N{0}_M{1}.json'.format(N,M)
    with open(file, 'w') as f:
        json.dump(results_dict, f)

if __name__ == "__main__":
    run_experiment()
