import torch as t
import torch.nn as nn
import alan
import numpy as np
import glob

from alan.utils import *
def generate_model(N,M,device=t.device('cpu'),ML=1, run=0):

    covariates = {}
    all_covariates = {}
#    sizes = {'plate_State': 51, 'plate_National_Polls':361, 'plate_State_Polls':1258, 'T':254, 'plate_P':161, 'plate_M':3, 'plate_Pop':3}

    for data in glob.glob( 'data/covariates/**.pt' ):
        name = data.split('/')[-1].split('.')[0]
        var = t.load('data/covariates/{}.pt'.format(name))
        all_var = None
        if var.shape[0] == 361:
            var = var.rename('plate_National_Polls',...)
        # if var.shape[0] == 51:
        #     var = var.rename('plate_State',...)
        if var.shape[0] == 1258:
            v = var
            var, test_var = t.chunk(v.clone(), 2, 0)
            var = var.rename('plate_State_Polls',...)
            all_var = v.rename('plate_State_Polls',...)
        if var.shape[0] == 254:
            var = var.rename('T',...)

        covariates[name] = var
        if all_var is not None:
            all_covariates[name] = all_var
        else:
            all_covariates[name] = var

    transform_data(covariates)
    transform_data(all_covariates)

    data = {}
    all_data = {}
    for d in glob.glob( 'data/**.pt' ):
        name = d.split('/')[-1].split('.')[0]
        var = t.load('data/{}.pt'.format(name))
        all_var = None
        # if var.shape[0] == 361:
        #     var = var.rename('plate_National_Polls',...)
        # if var.shape[0] == 51:
        #     var = var.rename('plate_State',...)
        if var.shape[0] == 1258:
            v = var
            var, test_var = t.chunk(v.clone(), 2, 0)
            var = var.rename('plate_State_Polls',...)
            all_var = v.rename('plate_State_Polls',...)
        # if var.shape[0] == 254:
        #     var = var.rename('T',...)

        data[name] = var
        if all_var is not None:
            all_data[name] = all_var
        else:
            all_data[name] = var


    N_national_polls = covariates.pop('N_national_polls')[0].int().item()   # Number of polls
    N_state_polls = covariates.pop('N_state_polls')[0].int().item() // 2 # Number of polls
    T = covariates.pop('T')[0].int().item()   # Number of days
    S = covariates.pop('S')[0].int().item()    # Number of states (for which at least 1 poll is available) + 1
    P_int = covariates.pop('P')[0].int().item()    # Number of pollsters
    M = covariates.pop('M')[0].int().item()    # Number of poll modes
    Pop = covariates.pop('Pop')[0].int().item()    # Number of poll populations

    N_national_polls = all_covariates.pop('N_national_polls')[0].int().item()   # Number of polls
    N_state_polls = all_covariates.pop('N_state_polls')[0].int().item() // 2 # Number of polls
    T = all_covariates.pop('T')[0].int().item()   # Number of days
    S = all_covariates.pop('S')[0].int().item()    # Number of states (for which at least 1 poll is available) + 1
    P_int = all_covariates.pop('P')[0].int().item()    # Number of pollsters
    M = all_covariates.pop('M')[0].int().item()    # Number of poll modes
    Pop = all_covariates.pop('Pop')[0].int().item()    # Number of poll populations

    sizes = {'plate_State': S, 'plate_National_Polls':N_national_polls, 'plate_State_Polls':N_state_polls,
             'T1':T, 'T2':T, 'plate_P':P_int, 'plate_M':M, 'plate_Pop':Pop}


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

        # tr('mu_b_T', alan.Normal(mu_b_prior, 0.01))
        def mu_b_transition(x):
            return alan.MultivariateNormal(x, ss_cov_mu_b_walk)

        # def mu_b_transition(x):
        #     return alan.Normal(x, 0.01)

        tr('mu_b', alan.Timeseries('mu_b_T', mu_b_transition), T="T1")

        tr('mu_c', alan.Normal(tr.zeros((P_int,)), sigma_c*tr.ones(())))

        tr('mu_m', alan.Normal(tr.zeros((M,)), sigma_m*tr.ones(())))

        tr('mu_pop', alan.Normal(tr.zeros((Pop,)), sigma_pop*tr.ones(())))

        tr('mu_e_bias', alan.Normal(tr.zeros(()), 0.02*tr.ones(())))

        tr('rho_e_bias', alan.Normal(0.7*tr.ones(()), 0.1*tr.ones(())))

        tr('e_bias', alan.Normal(tr['mu_e_bias'], (1/t.square(1-t.square(tr['rho_e_bias']))*tr.ones(()))))

        def e_transition(x):
            return alan.Normal(tr['rho_e_bias']*x, 0.02)

        tr('e', alan.Timeseries('e_bias', e_transition), T="T2")




        # tr('raw_measure_noise_national', alan.Normal(tr.zeros((N_national_polls,)), tr.ones(())))
        #
        tr('measure_noise_state', alan.Normal(tr.zeros(()), 0.04 * tr.ones(())), plates='plate_State_Polls')

        tr('polling_bias', alan.MultivariateNormal(tr.zeros((S,)), ss_cov_poll_bias))


        mu_b = tr['mu_b'].order(generic_dims(tr['mu_b'])[1:])[(day_state-1).long(), (state-1).long()]

        mu_c = tr['mu_c'][(poll_state-1).long()]

        mu_m = tr['mu_m'][(poll_mode_state-1).long()]

        mu_pop = tr['mu_pop'][(poll_pop_state - 1).long()]


        e = tr['e'].order(generic_dims(tr['e'])[1:])[(day_state-1).long()]

        polling_bias = tr['polling_bias'][(state - 1).long()]

        logit_pi_democrat_state =  mu_b + mu_c + mu_m + mu_pop + unadjusted_state * e + tr['measure_noise_state'] + polling_bias

        tr('n_democrat_state', alan.Binomial(n_two_share_state, logits=logit_pi_democrat_state))






    class Q(alan.AlanModule):
        def __init__(self):
            super().__init__()
            #raw_mu_b_T
            self.raw_mu_b_T_mean = nn.Parameter(t.zeros((S,)))
            self.raw_mu_b_T_scale = nn.Parameter(t.zeros((S,)))


            #raw_mu_b
            self.raw_mu_b_mean = nn.Parameter(t.zeros((T,S), names=('T1', None)))
            self.raw_mu_b_scale = nn.Parameter(t.zeros((T,S), names=('T1', None)))

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
            self.e_mean = nn.Parameter(t.zeros((T,), names=('T2',)))
            self.e_scale = nn.Parameter(t.zeros((T,), names=('T2',)))

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

            tr('measure_noise_state', alan.Normal(self.raw_measure_noise_state_mean, self.raw_measure_noise_state_scale.exp()))

            tr('polling_bias', alan.Normal(self.raw_polling_bias_mean, self.raw_polling_bias_scale.exp()))

    return P, Q, data, covariates, all_data, all_covariates, sizes

def transform_data(cov):
    national_cov_matrix_error_sd = t.sqrt(cov['state_weights'] @ cov['state_covariance_0'] @ cov['state_weights']);

    ## scale covariance
    cov['ss_cov_poll_bias'] = cov['state_covariance_0'] * t.square(cov['polling_bias_scale']/national_cov_matrix_error_sd);
    cov['ss_cov_mu_b_T'] = cov['state_covariance_0'] * t.square(cov['mu_b_T_scale']/national_cov_matrix_error_sd);
    cov['ss_cov_mu_b_walk'] = cov['state_covariance_0'] * t.square(cov['random_walk_scale']/national_cov_matrix_error_sd);
    ## transformation



if __name__ == '__main__':
    P, Q, data, covariates, all_data, all_covariates, sizes = generate_model(0,0)

    model = alan.Model(P, Q())
    #data_prior = model.sample_prior(platesizes = sizes, inputs = covariates)
    data = {'n_democrat_state': data['n_democrat_state']}#, 'n_democrat_national':data_prior['n_democrat_state']}

    all_data = {'n_democrat_state': all_data['n_democrat_state']}


    K = 3
    opt = t.optim.Adam(model.parameters(), lr=0.1)
    for j in range(200):
        opt.zero_grad()
        sample = model.sample_perm(K, data=data, inputs=covariates, reparam=True, device=t.device('cpu'))
        elbo = sample.elbo()
        (-elbo).backward()
        opt.step()

        print(elbo)

        for i in range(10):
            # try:
                # sample = model.sample_same(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))
            pred_likelihood = model.predictive_ll(sample, N = 10, data_all=all_data, inputs_all=all_covariates)
            #     break
            # except:
            #     pred_likelihood = 0
        print(pred_likelihood)
