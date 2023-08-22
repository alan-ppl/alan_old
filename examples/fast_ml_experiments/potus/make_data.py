import pandas as pd
import numpy as np
import torch as t
import re
import glob
from alan.experiment_utils import seed_torch
from pathlib import Path


Path("data/covariates").mkdir(parents=True, exist_ok=True)

r_csvs = result = glob.glob( 'r_csvs/**.csv' )

data_names = ['n_democrat_national', 'n_democrat_state']
for i in range(10):
    idx = t.randperm(1258)
    for csv in r_csvs:
        name = csv.split('/')[-1].split('.')[0]
        df = pd.read_csv(csv,index_col=0)
        tensor = t.tensor(df.values).squeeze(1)
        if tensor.shape[0] == 1258:
            tensor = tensor[idx]
        if name in data_names:
            t.save(tensor, 'data/{}_{}.pt'.format(name,i))
        else:
            t.save(tensor, 'data/covariates/{}_{}.pt'.format(name,i))
