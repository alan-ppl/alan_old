import torch as t
import torch.nn as nn
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
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

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

    P, Q, data, covariates, all_data, all_covariates = foo.generate_model(N,M,cfg.local, device)
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

            model = alan.Model(P, Q(), data = data, covariates = covariates)
            model.to(device)

            opt = t.optim.Adam(model.parameters(), lr=cfg.training.opt.lr)
            scheduler = t.optim.lr_scheduler.StepLR(opt, step_size=cfg.training.scheduler.step_size, gamma=cfg.training.scheduler.gamma)

            for j in range(cfg.training.num_iters):
                opt.zero_grad()
                objective = getattr(model, cfg.training.inference_method)
                obj = objective(K=K)
                (-obj).backward()
                opt.step()
                scheduler.step()
                per_seed_obj.append(obj.item())
                if 0 == j%1000:
                    print("Iteration: {0}, ELBO: {1:.2f}".format(j,obj.item()))


            objs.append(np.mean(per_seed_obj[-50:]))
            times.append((time.time() - start)/cfg.training.num_iters)
            if cfg.training.pred_ll.do_pred_ll and not cfg.local:
                pred_likelihood = model.predictive_ll(K = K, N = cfg.training.pred_ll.num_pred_ll_samples, data_all=all_data, covariates_all = all_covariates)
                pred_liks.append(pred_likelihood['obs'].item())
            else:
                pred_liks.append(0)
            ###
            # SAVING MODELS DOESN'T WORK YET
            ###
            if not os.path.exists(cfg.dataset + '/' + 'results/'):
                os.makedirs(cfg.dataset + '/' + 'results/')
            #
            # t.save(model.state_dict(), cfg.dataset + '/' + 'results/' + '{0}_{1}'.format(cfg.model, i))

        results_dict[N][M][K] = {'final_obj':np.mean(objs),'final_obj_std':np.std(objs), 'pred_likelihood':np.mean(pred_liks), 'pred_likelihood_std':np.std(pred_liks), 'objs': objs, 'pred_liks':pred_liks, 'avg_time':np.mean(times), 'std_time':np.std(times)}

    file = cfg.dataset + '/results/' + cfg.training.inference_method + ('_LIW_' if cfg.local else '_') + 'N{0}_M{1}.json'.format(N,M)
    with open(file, 'w') as f:
        json.dump(results_dict, f)

if __name__ == "__main__":
    run_experiment()
