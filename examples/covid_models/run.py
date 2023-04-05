import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

np.random.seed(123456)

import sys
import argparse
import datetime
import pickle
import alan
# import pymc3 as pm

from models.epi_params import EpidemiologicalParameters
from preprocessing.preprocess_mask_data import Preprocess_masks
from models.mask_models import (
    RandomWalkMobilityModel,
    MandateMobilityModel,
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

masks_object = Preprocess_masks(path)
masks_object.featurize(gatherings=GATHERINGS, masks=MASKS, smooth=SMOOTH, mobility=MOBI)
masks_object.make_preprocessed_object()
data = masks_object.data

all_observed_active, nRs, nDs, nCMs = model_data(masks_object.data)


ActiveCMs = masks_object.data.ActiveCMs
CMs = masks_object.data.CMs
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


if MODEL == "cases":
    del bd["deaths_delay_mean_mean"]
    del bd["deaths_delay_mean_sd"]
    del bd["deaths_delay_disp_mean"]
    del bd["deaths_delay_disp_sd"]





if MASKS == "wearing":
    P = RandomWalkMobilityModel(all_observed_active, nRs, nDs, nCMs, ActiveCMs, CMs,log_init_mean, log_init_sd)
    Q = RandomWalkMobilityModel(all_observed_active, nRs, nDs, nCMs, ActiveCMs, CMs,log_init_mean, log_init_sd, proposal=True)
elif MASKS == "mandate":
    P = MandateMobilityModel(all_observed_active, nRs, nDs, nCMs, ActiveCMs, CMs,log_init_mean, log_init_sd)
    Q = MandateMobilityModel(all_observed_active, nRs, nDs, nCMs, ActiveCMs, CMs,log_init_mean, log_init_sd, proposal=True)


MASS = "adapt_diag"  # Originally: 'jitter+adapt_diag'

r_walk_period = 7
nNP = int(nDs / r_walk_period) - 1


plate_sizes = {'plate_CM_alpha':nCMs - 2, 'plate_nRs':nRs,
               'Plate_obs':all_observed_active.shape[0],
               'plate_nNP':nNP}


model = alan.Model(P)
data = model.sample_prior(varnames='obs', platesizes=plate_sizes)

print(data)
print(data['obs'].shape)
print(all_observed_active.shape)
cond_model = alan.Model(P, Q()).condition(data=data)

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
