import torch as t
import torch.nn as nn
import alan
import numpy as np
import glob

from alan.utils import *
def generate_model(N,M,device=t.device('cpu'),ML=1, run=0):

    covariates = {}
#    sizes = {'plate_State': 51, 'plate_National_Polls':361, 'plate_State_Polls':1258, 'T':254, 'plate_P':161, 'plate_M':3, 'plate_Pop':3}

    for data in glob.glob( 'data/covariates/**.pt' ):
        name = data.split('/')[-1].split('.')[0]
        var = t.load('data/covariates/{}.pt'.format(name))
        # if var.shape[0] == 361:
        #     var = var.rename('plate_National_Polls',...)
        # # if var.shape[0] == 51:
        # #     var = var.rename('plate_State',...)
        # if var.shape[0] == 1258:
        #     var = var.rename('plate_State_Polls',...)
        # if var.shape[0] == 254:
        #     var = var.rename('T',...)

        covariates[name] = var

    transform_data(covariates)

    data = {}
    for d in glob.glob( 'data/**.pt' ):
        name = d.split('/')[-1].split('.')[0]
        var = t.load('data/{}.pt'.format(name))
        # if var.shape[0] == 361:
        #     var = var.rename('plate_National_Polls',...)
        # if var.shape[0] == 51:
        #     var = var.rename('plate_State',...)
        # if var.shape[0] == 1258:
        #     var = var.rename('plate_State_Polls',...)
        # if var.shape[0] == 254:
        #     var = var.rename('T',...)

        data[name] = var

    N_national_polls = covariates.pop('N_national_polls')[0].int().item()   # Number of polls
    N_state_polls = covariates.pop('N_state_polls')[0].int().item()   # Number of polls
    T = covariates.pop('T')[0].int().item()   # Number of days
    S = covariates.pop('S')[0].int().item()    # Number of states (for which at least 1 poll is available) + 1
    P_int = covariates.pop('P')[0].int().item()    # Number of pollsters
    M = covariates.pop('M')[0].int().item()    # Number of poll modes
    Pop = covariates.pop('Pop')[0].int().item()    # Number of poll populations

    print(M)
    sizes = {'plate_State': S, 'plate_National_Polls':N_national_polls, 'plate_State_Polls':N_state_polls,
             'T':T, 'plate_P':P_int, 'plate_M':M, 'plate_Pop':Pop}

    print(sizes)

    test_data, test_covariates, all_data, all_covariates = None, None, None, None

    def transformed_parameters(tr, state_weights,
                                   sigma_measure_noise_national,
                                   n_two_share_state,
                                   unadjusted_national,
                                   state,
                                   poll_mode_national,
                                   sigma_e_bias,
                                   N_national_polls,
                                   polling_bias_scale,
                                   mu_b_T_scale,
                                   day_national,
                                   poll_mode_state,
                                   poll_state,
                                   random_walk_scale,
                                   poll_pop_national,
                                   day_state,
                                   poll_national,
                                   sigma_measure_noise_state,
                                   n_two_share_national,
                                   sigma_c,
                                   unadjusted_state,
                                   mu_b_prior,
                                   sigma_m,
                                   state_covariance_0,
                                   poll_pop_state,
                                   sigma_pop,
                                   cholesky_ss_cov_poll_bias,
                                   cholesky_ss_cov_mu_b_T,
                                   cholesky_ss_cov_mu_b_walk):

            polling_bias = cholesky_ss_cov_poll_bias @ tr['raw_polling_bias']
            national_polling_bias_average = polling_bias @ state_weights

            mu_b = cholesky_ss_cov_mu_b_T @ tr['raw_mu_b_T'] + mu_b_prior

            mu_b = mu_b.repeat(T,1).T

            #REWRITE AS TIMESERIES
            for i in range(1,T):
                 mu_b[:,T - 1 - i] = cholesky_ss_cov_mu_b_walk @ tr['raw_mu_b'][:,T -1 - i] + mu_b[:, T - i];


            national_mu_b_average = state_weights @ mu_b;
            mu_c = tr['raw_mu_c'] * sigma_c
            mu_m = tr['raw_mu_m'] * sigma_m
            mu_pop = tr['raw_mu_pop'] * sigma_pop
            e_bias = t.zeros((T,))
            e_bias[0] = tr['raw_e_bias'][0] * sigma_e_bias;
            sigma_rho = t.sqrt(1-t.nn.functional.softmax(tr['rho_e_bias'])) * sigma_e_bias;
            for ti in range(1,T):
                e_bias[ti] = tr['mu_e_bias'] + tr['rho_e_bias'] * (e_bias[ti - 1] - tr['mu_e_bias']) + tr['raw_e_bias'][ti] * sigma_rho;
    #       //*** fill pi_democrat
            logit_pi_democrat_state = t.zeros((N_state_polls,))
            print(polling_bias)
            for i in range(N_state_polls):
                logit_pi_democrat_state[i] = mu_b[state[i]-1, day_state[i]-1] + \
                  mu_c[poll_state[i]-1] + \
                  mu_m[poll_mode_state[i]-1] + \
                  mu_pop[poll_pop_state[i]-1] + \
                  unadjusted_state[i] * e_bias[day_state[i]-1] + \
                  tr['raw_measure_noise_state'][i] * sigma_measure_noise_state + \
                  generic_order(polling_bias, generic_dims(polling_bias))[state[i]-1]

            logit_pi_democrat_national = national_mu_b_average[day_national-1] + \
            mu_c[poll_national-1] + \
            mu_m[poll_mode_national-1] + \
            mu_pop[poll_pop_national-1] + \
            unadjusted_national * e_bias[day_national-1] + \
            tr['raw_measure_noise_national'] * sigma_measure_noise_national + \
            national_polling_bias_average


            return logit_pi_democrat_state,  logit_pi_democrat_national

    def P(tr, state_weights,
              sigma_measure_noise_national,
              n_two_share_state,
              unadjusted_national,
              state,
              poll_mode_national,
              sigma_e_bias,
              polling_bias_scale,
              mu_b_T_scale,
              day_national,
              poll_mode_state,
              poll_state,
              random_walk_scale,
              poll_pop_national,
              day_state,
              poll_national,
              sigma_measure_noise_state,
              n_two_share_national,
              sigma_c,
              unadjusted_state,
              mu_b_prior,
              sigma_m,
              state_covariance_0,
              poll_pop_state,
              sigma_pop,
              cholesky_ss_cov_poll_bias,
              cholesky_ss_cov_mu_b_T,
              cholesky_ss_cov_mu_b_walk):
        '''
        Hierarchical Model
        '''

        tr('raw_mu_b_T', alan.Normal(tr.zeros((S,)), tr.ones(())))

        tr('raw_mu_b', alan.Normal(tr.zeros((S,T)), tr.ones(())))

        tr('raw_mu_c', alan.Normal(tr.zeros((P_int,)), tr.ones(())))

        tr('raw_mu_m', alan.Normal(tr.zeros((M,)), tr.ones(())))

        tr('raw_mu_pop', alan.Normal(tr.zeros((Pop,)), tr.ones(())))

        tr('mu_e_bias', alan.Normal(tr.zeros(()), 0.02*tr.ones(())))

        tr('rho_e_bias', alan.Normal(0.7*tr.ones(()), 0.1*tr.ones(())))

        tr('raw_e_bias', alan.Normal(tr.zeros((T,)), tr.ones(())))

        tr('raw_measure_noise_national', alan.Normal(tr.zeros((N_national_polls,)), tr.ones(())))

        tr('raw_measure_noise_state', alan.Normal(tr.zeros((N_state_polls,)), tr.ones(())))

        tr('raw_polling_bias', alan.Normal(tr.zeros((S,)), tr.ones(())))



        print(cholesky_ss_cov_mu_b_T @ tr['raw_mu_b_T'] + mu_b_prior)
        logit_pi_democrat_state, logit_pi_democrat_national = transformed_parameters(tr, state_weights,
                                       sigma_measure_noise_national,
                                       n_two_share_state,
                                       unadjusted_national,
                                       state,
                                       poll_mode_national,
                                       sigma_e_bias,
                                       N_national_polls,
                                       polling_bias_scale,
                                       mu_b_T_scale,
                                       day_national,
                                       poll_mode_state,
                                       poll_state,
                                       random_walk_scale,
                                       poll_pop_national,
                                       day_state,
                                       poll_national,
                                       sigma_measure_noise_state,
                                       n_two_share_national,
                                       sigma_c,
                                       unadjusted_state,
                                       mu_b_prior,
                                       sigma_m,
                                       state_covariance_0,
                                       poll_pop_state,
                                       sigma_pop,
                                       cholesky_ss_cov_poll_bias,
                                       cholesky_ss_cov_mu_b_T,
                                       cholesky_ss_cov_mu_b_walk)

        tr('n_democrat_state', alan.Binomial(n_two_share_state, logits=logit_pi_democrat_state))
        tr('n_democrat_national', alan.Binomial(n_two_share_national, logits=logit_pi_democrat_national))






    class Q(alan.AlanModule):
        def __init__(self):
            super().__init__()
            #sigma
            self.raw_mu_b_T_mean = nn.Parameter(t.zeros(()))
            self.raw_mu_b_T_scale = nn.Parameter(t.zeros(()))


            #theta
            self.theta_mean = nn.Parameter(t.zeros((M,), names=('plate_Players',)))
            self.theta_scale = nn.Parameter(t.zeros((M,), names=('plate_Players',)))


        def forward(self, tr, run_type, bus_company_name):
            #Year level

            tr('sigma', alan.Normal(self.raw_mu_b_T_mean, self.sigma_scale.exp()))
            tr('theta', alan.Normal(self.theta_mean, self.theta_scale.exp()))

    return P, Q, data, covariates, test_data, test_covariates, all_data, all_covariates, sizes

def transform_data(covariates):
    national_cov_matrix_error_sd = t.sqrt(covariates['state_weights'] @ covariates['state_covariance_0'] @ covariates['state_weights']);

    ## scale covariance
    ss_cov_poll_bias = covariates['state_covariance_0'] * t.square(covariates['polling_bias_scale']/national_cov_matrix_error_sd);
    ss_cov_mu_b_T = covariates['state_covariance_0'] * t.square(covariates['mu_b_T_scale']/national_cov_matrix_error_sd);
    ss_cov_mu_b_walk = covariates['state_covariance_0'] * t.square(covariates['random_walk_scale']/national_cov_matrix_error_sd);
    ## transformation

    names = ss_cov_poll_bias.names
    covariates['cholesky_ss_cov_poll_bias'] = t.linalg.cholesky(ss_cov_poll_bias.rename(None)).rename(names[0],...);
    names = ss_cov_mu_b_T.names
    covariates['cholesky_ss_cov_mu_b_T'] = t.linalg.cholesky(ss_cov_mu_b_T.rename(None)).rename(names[0],...);
    names = ss_cov_mu_b_walk.names
    covariates['cholesky_ss_cov_mu_b_walk'] = t.linalg.cholesky(ss_cov_mu_b_walk.rename(None)).rename(names[0],...);


if '__main__':
    P, Q, data, covariates, test_data, test_covariates, all_data, all_covariates, sizes = generate_model(0,0)

    model = alan.Model(P, Q())
    data_prior = model.sample_prior(platesizes = sizes, inputs = covariates)

    print(data_prior)
