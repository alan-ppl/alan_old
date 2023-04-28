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
        if var.shape[0] == 361:
            var = var.rename('plate_National_Polls',...)
        # if var.shape[0] == 51:
        #     var = var.rename('plate_State',...)
        if var.shape[0] == 1258:
            var = var.rename('plate_State_Polls',...)
        if var.shape[0] == 254:
            var = var.rename('T',...)

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
        if var.shape[0] == 1258:
            var = var.rename('plate_State_Polls',...)
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

    sizes = {'plate_State': S, 'plate_National_Polls':N_national_polls, 'plate_State_Polls':N_state_polls,
             'T':T, 'plate_P':P_int, 'plate_M':M, 'plate_Pop':Pop}

    print(sizes)
    test_data, test_covariates, all_data, all_covariates = None, None, None, None

    # def transformed_parameters(tr, state_weights,
    #                                sigma_measure_noise_national,
    #                                n_two_share_state,
    #                                unadjusted_national,
    #                                state,
    #                                poll_mode_national,
    #                                sigma_e_bias,
    #                                N_national_polls,
    #                                polling_bias_scale,
    #                                mu_b_T_scale,
    #                                day_national,
    #                                poll_mode_state,
    #                                poll_state,
    #                                random_walk_scale,
    #                                poll_pop_national,
    #                                day_state,
    #                                poll_national,
    #                                sigma_measure_noise_state,
    #                                n_two_share_national,
    #                                sigma_c,
    #                                unadjusted_state,
    #                                mu_b_prior,
    #                                sigma_m,
    #                                state_covariance_0,
    #                                poll_pop_state,
    #                                sigma_pop,
    #                                cholesky_ss_cov_poll_bias,
    #                                cholesky_ss_cov_mu_b_T,
    #                                cholesky_ss_cov_mu_b_walk):
    #
    #         polling_bias = cholesky_ss_cov_poll_bias @ tr['raw_polling_bias']
    #         national_polling_bias_average = polling_bias @ state_weights
    #
    #         mu_b = cholesky_ss_cov_mu_b_T @ tr['raw_mu_b_T'] + mu_b_prior
    #
    #         mu_b = mu_b.repeat(T,1).T
    #
    #         #REWRITE AS TIMESERIES
    #         for i in range(1,T):
    #              mu_b[:,T - 1 - i] = cholesky_ss_cov_mu_b_walk @ tr['raw_mu_b'][:,T -1 - i] + mu_b[:, T - i];
    #
    #
    #         national_mu_b_average = state_weights @ mu_b;
    #         mu_c = tr['raw_mu_c'] * sigma_c
    #         mu_m = tr['raw_mu_m'] * sigma_m
    #         mu_pop = tr['raw_mu_pop'] * sigma_pop
    #         e_bias = tr.zeros((T,))
    #         e_bias[0] = tr['raw_e_bias'][0] * sigma_e_bias;
    #         sigma_rho = t.sqrt(1-t.nn.functional.softmax(tr['rho_e_bias'])) * sigma_e_bias;
    #         for ti in range(1,T):
    #             e_bias[ti] = tr['mu_e_bias'] + tr['rho_e_bias'] * (e_bias[ti - 1] - tr['mu_e_bias']) + tr['raw_e_bias'][ti] * sigma_rho;
    # #       //*** fill pi_democrat
    #         logit_pi_democrat_state = t.zeros((N_state_polls,))
    #
    #         for i in range(N_state_polls):
    #             logit_pi_democrat_state[i] = mu_b[state[i]-1, day_state[i]-1] + \
    #               mu_c[poll_state[i]-1] + \
    #               mu_m[poll_mode_state[i]-1] + \
    #               mu_pop[poll_pop_state[i]-1] + \
    #               unadjusted_state[i] * e_bias[day_state[i]-1] + \
    #               tr['raw_measure_noise_state'][i] * sigma_measure_noise_state + \
    #               generic_order(polling_bias, generic_dims(polling_bias))[state[i]-1]
    #
    #         logit_pi_democrat_national = national_mu_b_average[day_national-1] + \
    #         mu_c[poll_national-1] + \
    #         mu_m[poll_mode_national-1] + \
    #         mu_pop[poll_pop_national-1] + \
    #         unadjusted_national * e_bias[day_national-1] + \
    #         tr['raw_measure_noise_national'] * sigma_measure_noise_national + \
    #         national_polling_bias_average
    #
    #
    #         return logit_pi_democrat_state,  logit_pi_democrat_national

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
              ss_cov_poll_bias,
              ss_cov_mu_b_T,
              ss_cov_mu_b_walk):
        '''
        Hierarchical Model
        '''

        tr('mu_b_T', alan.MultivariateNormal(mu_b_prior, ss_cov_mu_b_T))

        def mu_b_transition(x):
            return alan.MultivariateNormal(x, ss_cov_mu_b_walk)

        tr('mu_b', alan.Timeseries('mu_b_T', mu_b_transition), T="T")

        tr('mu_c', alan.Normal(tr.zeros((P_int,)), sigma_c*tr.ones(())))

        tr('mu_m', alan.Normal(tr.zeros((M,)), sigma_m*tr.ones(())))

        tr('mu_pop', alan.Normal(tr.zeros((Pop,)), sigma_pop*tr.ones(())))

        tr('mu_e_bias', alan.Normal(tr.zeros(()), 0.02*tr.ones(())))

        tr('rho_e_bias', alan.Normal(0.7*tr.ones(()), 0.1*tr.ones(())))

        tr('e_bias', alan.Normal(tr['mu_e_bias'], (1/t.square(1-t.square(tr['rho_e_bias']))*tr.ones(()))))

        def e_transition(x):
            return alan.Normal(tr['rho_e_bias']*x, 0.02)

        tr('e', alan.Timeseries('e_bias', e_transition), T="T")




        # tr('raw_measure_noise_national', alan.Normal(tr.zeros((N_national_polls,)), tr.ones(())))
        #
        tr('measure_noise_state', alan.Normal(tr.zeros(()), 0.04 * tr.ones(())), plates='plate_State_Polls')

        tr('polling_bias', alan.MultivariateNormal(tr.zeros((S,)), ss_cov_poll_bias))

        # tr('init_state_logit', alan.Normal(tr.zeros((S,)), tr.ones(())))
        # def n_democrat_state_transition(x, state, day_state, poll_state, poll_mode_state,
        #                                    poll_pop_state, unadjusted_state):
        #
        #     #mu_b = generic_order(tr['mu_b'], generic_dims(tr['mu_b']))#[:, state, day_state][generic_dims(tr['mu_b'])[0]]
        #     #print(mu_b.shape)
        #
        #     return alan.Normal(tr['mu_b'][day_state, state] + \
        #       mu_c[poll_state] + \
        #       mu_m[poll_mode_state] + \
        #       mu_pop[poll_pop_state] + \
        #       unadjusted_state * e_bias[day_state] + \
        #       generic_order(polling_bias, generic_dims(polling_bias))[state], sigma_measure_noise_state)


        mu_b = tr['mu_b'].order(generic_dims(tr['mu_b'])[1:])[(day_state-1).long(), (state-1).long()]

        mu_c = tr['mu_c'][(poll_state-1).long()]

        mu_m = tr['mu_m'][(poll_mode_state-1).long()]

        mu_pop = tr['mu_pop'][(poll_pop_state - 1).long()]

        e = tr['e'].order(generic_dims(tr['e'])[1:])[(day_state-1).long()]

        polling_bias = tr['polling_bias'][(state - 1).long()]

        logit_pi_democrat_state =  mu_b + mu_c + mu_m + mu_pop + unadjusted_state * e + tr['measure_noise_state'] + polling_bias
        # tr('logit_pi_democrat_state', alan.Timeseries('init_state_logit', n_democrat_state_transition, (state.long(), day_state.long(), poll_state.long(), poll_mode_state.long(),
        #                                    poll_pop_state.long(), unadjusted_state.long())), T='plate_State_Polls')
        #print(logit_pi_democrat_state)
        tr('n_democrat_state', alan.Binomial(n_two_share_state, logits=logit_pi_democrat_state))
        #tr('n_democrat_national', alan.Binomial(n_two_share_national, logits=logit_pi_democrat_national))






    class Q(alan.AlanModule):
        def __init__(self):
            super().__init__()
            #raw_mu_b_T
            self.raw_mu_b_T_mean = nn.Parameter(t.zeros((S,)))
            self.raw_mu_b_T_scale = nn.Parameter(t.zeros((S,)))


            #raw_mu_b
            self.raw_mu_b_mean = nn.Parameter(t.zeros((T,S), names=('T', None)))
            self.raw_mu_b_scale = nn.Parameter(t.zeros((T,S), names=('T', None)))

            #raw_mu_c
            self.raw_mu_c_mean = nn.Parameter(t.zeros((P_int,)))
            self.raw_mu_c_scale = nn.Parameter(t.zeros((P_int,)))

            #raw_mu_m
            self.raw_mu_m_mean = nn.Parameter(t.zeros((M,)))
            self.raw_mu_m_scale = nn.Parameter(t.zeros((M,)))

            #raw_mu_pop
            self.raw_mu_pop_mean = nn.Parameter(t.zeros((Pop,)))
            self.raw_mu_pop_scale = nn.Parameter(t.zeros((Pop,)))

            #mu_e_bias
            self.mu_e_bias_mean = nn.Parameter(t.zeros(()))
            self.mu_e_bias_scale = nn.Parameter(t.zeros(()))

            #rho_e_bias
            self.rho_e_bias_mean = nn.Parameter(t.zeros(()))
            self.rho_e_bias_scale = nn.Parameter(t.zeros(()))

            #raw_e_bias
            self.raw_e_bias_mean = nn.Parameter(t.zeros(()))
            self.raw_e_bias_scale = nn.Parameter(t.zeros(()))

            #e
            self.e_mean = nn.Parameter(t.zeros((T,), names=('T',)))
            self.e_scale = nn.Parameter(t.zeros((T,), names=('T',)))

            #raw_measure_noise_national
            self.raw_measure_noise_national_mean = nn.Parameter(t.zeros((N_national_polls,)))
            self.raw_measure_noise_national_scale = nn.Parameter(t.zeros((N_national_polls,)))

            #raw_measure_noise_national
            self.raw_measure_noise_state_mean = nn.Parameter(t.zeros((N_state_polls,), names=('plate_State_Polls',)))
            self.raw_measure_noise_state_scale = nn.Parameter(t.zeros((N_state_polls,), names=('plate_State_Polls',)))

            #raw_polling_bias
            self.raw_polling_bias_mean = nn.Parameter(t.zeros((S,)))
            self.raw_polling_bias_scale = nn.Parameter(t.zeros((S,)))

        def forward(self, tr, state_weights,
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
                  ss_cov_poll_bias,
                  ss_cov_mu_b_T,
                  ss_cov_mu_b_walk):
            #Year level

            tr('mu_b_T', alan.Normal(self.raw_mu_b_T_mean, self.raw_mu_b_T_scale.exp()))

            tr('mu_b', alan.Normal(self.raw_mu_b_mean, self.raw_mu_b_scale.exp()))

            tr('mu_c', alan.Normal(self.raw_mu_c_mean, self.raw_mu_c_scale.exp()))

            tr('mu_m', alan.Normal(self.raw_mu_m_mean, self.raw_mu_m_scale.exp()))

            tr('mu_pop', alan.Normal(self.raw_mu_pop_mean, self.raw_mu_pop_scale.exp()))

            tr('mu_e_bias', alan.Normal(self.mu_e_bias_mean, self.mu_e_bias_scale.exp()))

            tr('rho_e_bias', alan.Normal(self.rho_e_bias_mean, self.rho_e_bias_scale.exp()))

            tr('e_bias', alan.Normal(self.raw_e_bias_mean, self.raw_e_bias_scale.exp()))

            tr('e', alan.Normal(self.e_mean, self.e_scale.exp()))

            #tr('measure_noise_national', alan.Normal(self.raw_measure_noise_national_mean, self.raw_measure_noise_national_scale.exp()))

            tr('measure_noise_state', alan.Normal(self.raw_measure_noise_state_mean, self.raw_measure_noise_state_scale.exp()))

            tr('polling_bias', alan.Normal(self.raw_polling_bias_mean, self.raw_polling_bias_scale.exp()))

    return P, Q, data, covariates, test_data, test_covariates, all_data, all_covariates, sizes

def transform_data(covariates):
    national_cov_matrix_error_sd = t.sqrt(covariates['state_weights'] @ covariates['state_covariance_0'] @ covariates['state_weights']);

    ## scale covariance
    covariates['ss_cov_poll_bias'] = covariates['state_covariance_0'] * t.square(covariates['polling_bias_scale']/national_cov_matrix_error_sd);
    covariates['ss_cov_mu_b_T'] = covariates['state_covariance_0'] * t.square(covariates['mu_b_T_scale']/national_cov_matrix_error_sd);
    covariates['ss_cov_mu_b_walk'] = covariates['state_covariance_0'] * t.square(covariates['random_walk_scale']/national_cov_matrix_error_sd);
    ## transformation



if '__main__':
    P, Q, data, covariates, test_data, test_covariates, all_data, all_covariates, sizes = generate_model(0,0)

    model = alan.Model(P, Q())
    #data_prior = model.sample_prior(platesizes = sizes, inputs = covariates)
    data = {'n_democrat_state': data['n_democrat_state']}#, 'n_democrat_national':data_prior['n_democrat_state']}
    print(data)
    K = 3
    sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))
    elbo = sample.elbo().item()

    print(elbo)
