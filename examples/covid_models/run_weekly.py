import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch as t
np.random.seed(123456)

import sys
import argparse
import datetime
import pickle
import alan
# import pymc3 as pm

from models.epi_params import EpidemiologicalParameters
from preprocessing.preprocess_mask_data import Preprocess_masks
from models.mask_models_weekly import (
    RandomWalkMobilityModel,
    model_data
)

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", dest="model", type=str, help="Model type")
argparser.add_argument("--masks", dest="masks", type=str, help="Which mask feature")
argparser.add_argument(
    "--w_par", dest="w_par", type=str, help="Which wearing parameterisation"
)
argparser.add_argument("--mob", dest="mob", type=str, help="How to include mobility")

# argparser.add_argument('--filter', dest='filtered', type=str, help='How to remove regions')
# argparser.add_argument('--gatherings', dest='gatherings', type=int, help='how many gatherings features')
argparser.add_argument("--tuning", dest="tuning", type=int, help="tuning samples")
argparser.add_argument("--draws", dest="draws", type=int, help="draws")
argparser.add_argument("--chains", dest="chains", type=int, help="chains")
# argparser.add_argument('--hide_ends', dest='hide_ends', type=str)
args, _ = argparser.parse_known_args()

MODEL = args.model
MASKS = args.masks
W_PAR = args.w_par if args.w_par else "exp"
MOBI = args.mob
TUNING = args.tuning if args.tuning else 1000
DRAWS = args.draws if args.draws else 500
CHAINS = args.chains if args.chains else 4
# FILTERED = args.filtered

US = True
SMOOTH = False
GATHERINGS = 3  # args.gatherings if args.gatherings else 3
# MASKING = True # Always true


# prep data object
path = f"data/modelling_set/master_data_mob_{MOBI}_us_{US}_m_w.csv"
print(path)
masks_object = Preprocess_masks(path)
masks_object.featurize(gatherings=GATHERINGS, masks=MASKS, smooth=SMOOTH, mobility=MOBI)
masks_object.make_preprocessed_object()
data = masks_object.data

all_observed_active, nRs, nDs, nCMs = model_data(masks_object.data)



ActiveCMs = np.add.reduceat(masks_object.data.ActiveCMs, np.arange(0, nDs, 7), 2)

print(ActiveCMs.shape)
ActiveCMs = t.from_numpy(np.moveaxis(ActiveCMs,[0,2,1], [0,1,2]))
print(ActiveCMs.shape)
CMs = masks_object.data.CMs

#Number of weeks
nWs = int(np.ceil(nDs/7))


print('nRs')
print(nRs)

print('nDs')
print(nDs)

print('nCMs')
print(nCMs)

print('nWs')
print(nWs)

# model specification
ep = EpidemiologicalParameters()
bd = ep.get_model_build_dict()



def set_init_infections(data, d):
    n_masked_days = 10
    first_day_new = data.NewCases[:, n_masked_days]
    first_day_new = first_day_new[first_day_new.mask == False]
    median_init_size = np.median(first_day_new)


    if median_init_size == 0:
        median_init_size = 50

    return np.log(median_init_size), np.log(median_init_size)


log_init_mean, log_init_sd = set_init_infections(data, bd)

bd["wearing_parameterisation"] = W_PAR



r_walk_period = 7
nNP = int(nDs / r_walk_period) - 1


plate_sizes = {'plate_nRs':nRs,
               'nWs':nWs}

#New weekly cases
newcases_weekly = np.add.reduceat(data.NewCases, np.arange(0, nDs, 7), 1)
newcases_weekly = t.from_numpy(newcases_weekly).rename('plate_nRs', 'nWs' )
#NPI active CMs
ActiveCMs_NPIs = ActiveCMs[:, :, :-2].rename('plate_nRs', 'nWs', None)

ActiveCMs_wearing = ActiveCMs[:, :, -1].rename('plate_nRs', 'nWs' )
ActiveCMs_mobility = ActiveCMs[:, :, -2].rename('plate_nRs', 'nWs')

covariates = {'ActiveCMs_NPIs':ActiveCMs_NPIs, 'ActiveCMs_wearing':ActiveCMs_wearing, 'ActiveCMs_mobility':ActiveCMs_mobility}
if MASKS == "wearing":
    P = RandomWalkMobilityModel(nRs, nWs, nCMs, CMs,log_init_mean, log_init_sd)
    Q = RandomWalkMobilityModel_Q(nRs, nWs, nCMs, CMs,log_init_mean, log_init_sd, proposal=True)
elif MASKS == "mandate":
    P = MandateMobilityModel(nRs, nWs, nCMs, CMs, log_init_mean, log_init_sd)
    Q = MandateMobilityModel(nRs, nWs, nCMs, CMs, log_init_mean, log_init_sd, proposal=True)





model = alan.Model(P)
data = model.sample_prior(varnames='obs', inputs=covariates, platesizes=plate_sizes)
print(data)
print(newcases_weekly.shape)
cond_model = alan.Model(P, Q).condition(data={'obs':newcases_weekly}, inputs=covariates)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=10
print("K={}".format(K))
for i in range(20000):
  opt.zero_grad()
  elbo = cond_model.sample_perm(K, True).elbo()
  (-elbo).backward()
  opt.step()

  if 0 == i%1000:
      print(elbo.item())


dt = datetime.datetime.now().strftime("%m-%d-%H:%M")

if MASKS == "wearing":
    idstr = f"pickles/{MASKS}_{W_PAR}_{MODEL}_{len(data.Rs)}_{MOBI}_{dt}"
else:
    idstr = f"pickles/{MASKS}_{MODEL}_{len(data.Rs)}_{MOBI}_{dt}"

pickle.dump(model.trace, open(idstr + ".pkl", "wb"))

with open(idstr + "_cols", "w") as f:
    f.write(", ".join(data.CMs))
