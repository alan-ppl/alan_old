import pandas as pd
import numpy as np
import torch as t
import re
import glob
from alan.experiment_utils import seed_torch


r_csvs = result = glob.glob( 'r_csvs/**.csv' )

for csv in r_csvs:
    name = csv.split('/')[-1].split('.')[0]
    df = pd.read_csv(csv,index_col=0)
    tensor = t.tensor(df.values).squeeze(1)
    t.save(tensor, 'data/{}.pt'.format(name))
