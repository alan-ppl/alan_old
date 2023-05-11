import torch as t
import torch.nn as nn
import alan
import numpy as np

def generate_model(N,M,device,ML=1, run=0):


    sizes = {'plate_Players': M, 'plate_Match':N}

    covariates = None
    test_covariates = None
    all_covariates = None

    data = {'obs':t.load('bus_breakdown/data/delay_train_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
    test_data = {'obs':t.load('bus_breakdown/data/delay_test_{}.pt'.format(run)).rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']],-2)}


    def P(tr):
      '''
      Hierarchical Model
      '''

      #Player level
      tr('sigma', alan.HalfNormal(tr.ones(())))
      tr('theta', alan.Normal(tr.zeros(()), tr['sigma']), plates = 'plate_Players')

      pairwise_diff_2 = generic_order(tr['theta'], generic_dims(tr['theta']))
      pairwise_diff_1 = pairwise_diff_2.unsqueeze(-1)


      logits = (pairwise_diff_1 - pairwise_diff_2)[generic_dims(tr['theta']) + [None]]
      tr('obs', alan.Bernoulli(logits=logits), plates='plate_Match')




    if ML == 1:
        class Q(alan.AlanModule):
            def __init__(self):
                super().__init__()
                #sigma_beta
                self.sigma = alan.MLHalfNormal()
                #mu_beta
                self.theta = alan.MLNormal({'plate_Players': M})


            def forward(self, tr, run_type, bus_company_name):
                #Year level

                tr('sigma', self.sigma())
                tr('theta', self.theta())
    elif ML == 2:
        class Q(alan.AlanModule):
            def __init__(self):
                super().__init__()
                #sigma_beta
                self.sigma = alan.ML2HalfNormal()
                #mu_beta
                self.theta = alan.ML2Normal({'plate_Players': M})


            def forward(self, tr, run_type, bus_company_name):
                #Year level

                tr('sigma', self.sigma())
                tr('theta', self.theta())

    return P, Q, data, covariates, test_data, test_covariates, all_data, all_covariates, sizes
