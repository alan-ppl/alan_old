import torch as t
import torch.nn as nn
import alan
import numpy as np
import glob

def generate_model(N,M,device,ML=1, run=0):

    covariates = {}
    sizes = {'plate_State': 51, 'plate_National_Polls':361, 'plate_State_Polls':1258, 'T':254}

    for data in glob.glob( 'data/covariates/**.pt' ):
        name = csv.split('/')[-1].split('.')[0]
        var = t.load('data/{}.pt'.format(name))
        if var.shape[0] == 361:
            var = var.rename('plate_National_Polls',...)
        if var.shape[0] == 51:
            var = var.rename('plate_State',...)
        if var.shape[0] == 1258:
            var = var.rename('plate_State_Polls',...)
        if var.shape[0] == 254:
            var = var.rename('T',...)

        covariates[name] = var

        transform_data(covariates)

    data = {}
    for data in glob.glob( 'data/**.pt' ):
        name = csv.split('/')[-1].split('.')[0]
        var = t.load('data/{}.pt'.format(name))
        if var.shape[0] == 361:
            var = var.rename('plate_National_Polls',...)
        if var.shape[0] == 51:
            var = var.rename('plate_State',...)
        if var.shape[0] == 1258:
            var = var.rename('plate_State_Polls',...)
        if var.shape[0] == 254:
            var = var.rename('T',...)

        data[name] = var

    def P(tr, covariates):
        '''
        Hierarchical Model
        '''

        tr('raw_mu_b_T', alan.Normal(tr.zeros((51,)), tr.ones(())))

        tr('raw_mu_b', alan.Normal(tr.zeros((51,254)), tr.ones(())))

        tr('raw_mu_c', alan.Normal(tr.zeros(()), tr.ones(())))

        tr('raw_mu_m', alan.Normal(tr.zeros(()), tr.ones(())))

        tr('raw_mu_pop', alan.Normal(tr.zeros(()), tr.ones(())))

        tr('mu_e_bias', alan.Normal(tr.zeros(()), 0.02*tr.ones(())))

        tr('rho_e_bias', alan.Normal(0.7*tr.ones(()), 0.1*tr.ones(())))

        tr('raw_e_bias', alan.Normal(tr.zeros(()), tr.ones(())))

        tr('raw_measure_noise_nationals', alan.Normal(tr.zeros(()), tr.ones(())))

        tr('raw_measure_noise_state', alan.Normal(tr.zeros(()), tr.ones(())))

        tr('raw_polling_bias', alan.Normal(tr.zeros(()), tr.ones(())))

        # def transformed parameters(tr):
        #       # //*** parameters
        #       # matrix[S, T] mu_b;
        #       # vector[P] mu_c;
        #       # vector[M] mu_m;
        #       # vector[Pop] mu_pop;
        #       # vector[T] e_bias;
        #       # vector[S] polling_bias = cholesky_ss_cov_poll_bias * raw_polling_bias;
        #       # vector[T] national_mu_b_average;
        #       # real national_polling_bias_average = transpose(polling_bias) * state_weights;
        #       # real sigma_rho;
        #       # //*** containers
        #       # vector[N_state_polls] logit_pi_democrat_state;
        #       # vector[N_national_polls] logit_pi_democrat_national;
        #       //*** construct parameters
        #       mu_b[:,T] = cholesky_ss_cov_mu_b_T * raw_mu_b_T + mu_b_prior;  // * mu_b_T_model_estimation_error
        #       mu_b
        #       for (i in 1:(T-1)) mu_b[:, T - i] = cholesky_ss_cov_mu_b_walk * raw_mu_b[:, T - i] + mu_b[:, T + 1 - i];
        #       national_mu_b_average = transpose(mu_b) * state_weights;
        #       mu_c = raw_mu_c * sigma_c;
        #       mu_m = raw_mu_m * sigma_m;
        #       mu_pop = raw_mu_pop * sigma_pop;
        #       e_bias[1] = raw_e_bias[1] * sigma_e_bias;
        #       sigma_rho = sqrt(1-square(rho_e_bias)) * sigma_e_bias;
        #       for (t in 2:T) e_bias[t] = mu_e_bias + rho_e_bias * (e_bias[t - 1] - mu_e_bias) + raw_e_bias[t] * sigma_rho;
        #       //*** fill pi_democrat
        #       for (i in 1:N_state_polls){
        #         logit_pi_democrat_state[i] =
        #           mu_b[state[i], day_state[i]] +
        #           mu_c[poll_state[i]] +
        #           mu_m[poll_mode_state[i]] +
        #           mu_pop[poll_pop_state[i]] +
        #           unadjusted_state[i] * e_bias[day_state[i]] +
        #           raw_measure_noise_state[i] * sigma_measure_noise_state +
        #           polling_bias[state[i]];
        #       }
        #       logit_pi_democrat_national =
        #         national_mu_b_average[day_national] +
        #         mu_c[poll_national] +
        #         mu_m[poll_mode_national] +
        #         mu_pop[poll_pop_national] +
        #         unadjusted_national .* e_bias[day_national] +
        #         raw_measure_noise_national * sigma_measure_noise_national +
        #         national_polling_bias_average;
        #     }





    class Q(alan.AlanModule):
        def __init__(self):
            super().__init__()
            #sigma
            self.sigma_scale = nn.Parameter(t.zeros(()))


            #theta
            self.theta_mean = nn.Parameter(t.zeros((M,), names=('plate_Players',)))
            self.theta_scale = nn.Parameter(t.zeros((M,), names=('plate_Players',)))


        def forward(self, tr, run_type, bus_company_name):
            #Year level

            tr('sigma', alan.HalfNormal(self.sigma_scale.exp()))
            tr('theta', alan.Normal(self.theta_mean, self.theta_scale.exp()))

    return P, Q, data, covariates, test_data, test_covariates, all_data, all_covariates, sizes

def transform_data(covariates):
    real national_cov_matrix_error_sd = sqrt(covariates['state_weights'].T @ covariates['state_covariance_0'] @ covariates['state_weights']);

    ## scale covariance
    ss_cov_poll_bias = covariates['state_covariance_0'] * t.square(covariates['polling_bias_scale']/covariates['national_cov_matrix_error_sd']);
    ss_cov_mu_b_T = covariates['state_covariance_0'] * t.square(covariates['mu_b_T_scale']/covariates['national_cov_matrix_error_sd']);
    ss_cov_mu_b_walk = covariates['state_covariance_0'] * t.square(covariates['random_walk_scale']/covariates['national_cov_matrix_error_sd']);
    ## transformation
    covariates['cholesky_ss_cov_poll_bias'] = t.cholesky(ss_cov_poll_bias);
    covariates['cholesky_ss_cov_mu_b_T'] = t.cholesky(ss_cov_mu_b_T);
    covariates['cholesky_ss_cov_mu_b_walk'] = t.cholesky(ss_cov_mu_b_walk);
