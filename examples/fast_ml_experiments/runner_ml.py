import torch as t
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
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



        P, Q, data, covariates, all_data, all_covariates, sizes = foo.generate_model(N,M, device, cfg.training.ML, i, cfg.use_data)

        m1 = alan.Model(P, Q).condition(data=data)

        samp = m1.sample_same(K, inputs=covariates, reparam=False, device=device)

        scales = {k:np.zeros((cfg.training.num_runs,cfg.training.num_iters), dtype=np.float32) for k in samp.weights().keys()}
        weights = {k:np.zeros((cfg.training.num_runs,cfg.training.num_iters), dtype=np.float32) for k in samp.weights().keys()}

        times = np.zeros((cfg.training.num_runs,cfg.training.num_iters), dtype=np.float32)
        non_zero_weights = np.zeros((cfg.training.num_runs,cfg.training.num_iters), dtype=np.float32)
        nans = np.asarray([0]*cfg.training.num_runs)
        for i in range(cfg.training.num_runs):
            seed_torch(i)
            P, Q, data, covariates, all_data, all_covariates, sizes = foo.generate_model(N,M, device, cfg.training.ML, i, cfg.use_data)

            if not cfg.use_data:
                if not cfg.dataset == 'potus':
                    data = {'obs':data.pop('obs')}
                else:
                    data = {'n_democrat_state':data.pop('n_democrat_state')}


            model = alan.Model(P, Q())
            model.to(device)

            
            for j in range(cfg.training.num_iters):

                if cfg.training.decay is not None:
                    lr = (j + 10)**(-cfg.training.decay)
                else:
                    lr = cfg.training.lr

                if t.cuda.is_available():
                    t.cuda.synchronize()
                start = time.time()
                sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=device)
                elbo = sample.elbo().item()
                per_seed_obj[i,j] = (elbo)
                model.update(lr, sample)

                if t.cuda.is_available():
                    t.cuda.synchronize()

                times[i,j] = (time.time() - start)

                #Predictive Log Likelihoods
                if cfg.training.pred_ll.do_pred_ll:
                    success=False
                    for k in range(10):
                        try:
                            sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=device)
                            pred_likelihood = model.predictive_ll(sample, N = cfg.training.pred_ll.num_pred_ll_samples, data_all=all_data, inputs_all=all_covariates)
                            if not cfg.dataset == 'potus':
                                pred_liks[i,j] = pred_likelihood['obs'].item()
                            else:
                                pred_liks[i,j] = pred_likelihood['n_democrat_state'].item()
                            success=True
                        except:
                            print('nan pred likelihood!')
                            nans[i] += 1
                        if success:
                            break
                    if not success:
                        pred_liks[i,j] = np.nan


                non_zero_weight = 0  
                num_latents = 0
                    
                for k,v in sample.weights().items():
                    scales[k][i,j] = q.__getattr__(k).mean2conv(*q.__getattr__(k).named_means)['scale'].mean()
                    weights[k][i,j] = (v[1].rename(None) > 0.001).sum()

                    non_zero_weight += (v[1].rename(None) > 0.001).sum()
                    num_latents += v[0].numel()

                non_zero_weights[i,j] = non_zero_weight



                if j % 100 == 0:
                    print("Iteration: {0}, ELBO: {1:.2f}, Predll: {2:.2f}".format(j,elbo,pred_liks[i,j]))


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
                                 'scales':scales,
                                 'weights':weights,
                                 'non_zero_weights':non_zero_weights,}


        file = cfg.dataset + '/results/' + cfg.model + '/ML_{}'.format(cfg.training.num_iters) + '_{}_'.format(cfg.training.lr) + 'K{0}_{1}.pkl'.format(K,cfg.use_data)
        with open(file, 'wb') as f:
            pickle.dump(results_dict, f)


if __name__ == "__main__":
    run_experiment()
