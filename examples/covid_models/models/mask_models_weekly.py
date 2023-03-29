"""
Original Authors: Gavin Leech and Charlie Rogers Smith
Source: https://raw.githubusercontent.com/g-leech/masks_v_mandates/main/epimodel/pymc3_models/mask_models.py
"""

import torch as t
import torch.nn as nn
import alan
import numpy as np
import math

from alan.utils import *
from .TruncatedNormal import TruncatedNormal
from .epi_params import EpidemiologicalParameters
# from epimodel.pymc3_distributions.asymmetric_laplace import AsymmetricLaplace


alan.new_dist("TruncatedNormal", TruncatedNormal, 0, {'loc': 0, 'scale': 0, "a":0, "b":0})

def model_data(data):
    """
    Extract useful information for the models from the data
    i.e nRs, nDs
    """
    observed_active = []
    nRs = len(data.Rs)
    nDs = len(data.Ds)
    nCMs = len(data.CMs)
    for r in range(nRs):
        for d in range(nDs):
            # if its not masked, after the cut, and not before 100 confirmed
            if (
                data.NewCases.mask[r, d] == False
                # and d > self.CMDelayCut
                and not np.isnan(data.Confirmed.data[r, d])
            ):
                observed_active.append(r * nDs + d)
            else:
                data.NewCases.mask[r, d] = True

    all_observed_active = t.tensor(observed_active)
    return all_observed_active, nRs, nDs, nCMs

class RandomWalkMobilityModel(nn.Module):
    def __init__(self,
        nRs,
        nDs,
        nCMs,
        CMs,
        log_init_mean,
        log_init_sd,
        proposal=False):
        """
        Constructor.

        """
        super().__init__()
        self.nRs = t.tensor(nRs)
        self.nDs = t.tensor(nDs)
        self.nCMs = t.tensor(nCMs)
        self.CMs = CMs
        self.proposal=proposal
        self.log_init_mean = log_init_mean
        self.log_init_sd = log_init_sd


    def forward(
        self,
        tr,
        ActiveCMs_NPIs,
        ActiveCMs_wearing,
        ActiveCMs_mobility,
        r_walk_period=7,
        r_walk_noise_scale_prior=0.15,
        intervention_prior="Gauss",
        cm_prior_scale=1,
        wearing_parameterisation="exp",
        wearing_mean=0,
        wearing_mean_linear=0,
        wearing_mean_quadratic=0,
        wearing_sigma=0.4,
        wearing_sigma_linear=0.26,
        wearing_sigma_quadratic=0.13,
        mobility_mean=1.704,
        mobility_sigma=0.44,
        R_prior_mean_mean=1.07,
        R_prior_mean_scale=0.2,
        R_noise_scale=0.4,
        cm_prior="skewed",
        gi_mean_mean=5.06,
        gi_mean_sd=0.33,
        gi_sd_mean=2.11,
        gi_sd_sd=0.5,
        cases_delay_mean_mean=10.92,
        cases_delay_disp_mean=5.41,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        mobility_leaveout=False,
        mob_and_wearing_only=False,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        # Ensure mobility feature is in the right place
        mob_feature = "avg_mobility_no_parks_no_residential"
        assert self.CMs[-2] == mob_feature


        # tr('CM_Alpha', alan.Normal(
        #     tr.zeros((self.nCMs -2,)), cm_prior_scale)
        # )
        tr('CM_Alpha', alan.Normal(
            tr.zeros((self.nCMs-2,)), cm_prior_scale))


        # self.CMReduction = (-1.0) * tr['CM_Alpha']

        tr("Wearing_Alpha", alan.Normal(wearing_mean, wearing_sigma)
        )
        # self.WearingReduction = t.exp((-1.0) * tr['Wearing_Alpha'])

        tr('Mobility_Alpha', alan.Normal(
            mobility_mean, mobility_sigma),
        )
        # self.MobilityReduction = (2.0 * (t.exp(-1.0 * tr['Mobility_Alpha']))) / (1.0 + t.exp(-1.0 * tr['Mobility_Alpha']))

        # tr("HyperRMean", alan.dist.TruncatedNormal(
        #     R_prior_mean_mean, R_prior_mean_scale, a=0.1
        # ))
        #
        # tr("HyperRVar", alan.HalfNormal(R_noise_scale))

        tr("RegionR", alan.Normal(R_prior_mean_mean, R_prior_mean_scale + R_noise_scale))
        # # tr("RegionR_noise", alan.Normal(0, 0.1), plates='plate_nRs')
        # # tr("RegionR_noise", alan.Normal(tr.zeros((self.nRs,)), 1))
        # RegionR = tr['HyperRMean'] + tr['RegionR_noise'] * tr['HyperRVar']


        # load CMs active without wearing, compute log-R reduction and region log-R based on NPIs active

        ActiveCMReduction = tr['CM_Alpha'] * ActiveCMs_NPIs

        #Write script to do this quickly
        #growth_reduction = t.sum(generic_order(self.ActiveCMReduction, generic_dims(self.ActiveCMReduction))[generic_dims(self.ActiveCMReduction)[:-1]], axis=-1)
        growth_reduction = t.sum(ActiveCMReduction)
        # calculating reductions for each of the wearing parameterisations

        ActiveCMReduction_wearing = tr['Wearing_Alpha'] * ActiveCMs_wearing

        growth_reduction_wearing = ActiveCMReduction_wearing




        # make reduction for mobility

        ActiveCMReduction_mobility = tr['Mobility_Alpha'] * ActiveCMs_mobility

        growth_reduction_mobility = -1.0 * t.log(
                (2.0 * t.exp(-1.0 * ActiveCMReduction_mobility))
                / (1.0 + t.exp(-1.0 * ActiveCMReduction_mobility)),
        )


        initial_mobility_reduction = generic_order(growth_reduction_mobility, generic_dims(growth_reduction_mobility)).select(-1,0)[generic_dims(growth_reduction_mobility)[:-1]]

        # tr("r_walk_noise_scale", alan.HalfNormal(r_walk_noise_scale_prior))


        ExpectedLogR = tr['RegionR'] \
            - growth_reduction \
            - growth_reduction_wearing \
            - (growth_reduction_mobility - initial_mobility_reduction) \
            #+ tr['r_walk_noise']


        # convert R into growth rates
        tr('GI_mean', alan.Normal(np.log(gi_mean_mean), gi_mean_sd))
        tr("GI_sd", alan.Normal(gi_sd_mean, gi_sd_sd))

        gi_beta = t.exp(tr['GI_mean'] / tr['GI_sd'] ** 2)
        gi_alpha = tr['GI_mean'] ** 2 / tr['GI_sd'] ** 2


        tr("InitialSize_log", alan.Normal(t.tensor(self.log_init_mean), self.log_init_sd), plates='plate_nRs')

        def transition(x, inputs):
            ExpectedLogR = inputs
            ExpectedGrowth = gi_beta + (
                    (ExpectedLogR / gi_alpha)
                    - t.log(t.ones_like(ExpectedLogR))
                )
            return alan.Normal(x + ExpectedGrowth, 1)

        #print(t.exp(ExpectedLogR))
        tr('Log_Infected', alan.Timeseries("InitialSize_log", transition, t.exp(ExpectedLogR)), T="nWs")
        Infected = t.exp(tr['Log_Infected'])

        tr('psi', alan.Normal(0,np.log(45)))
        # effectively handle missing values ourselves
        # likelihood
        r = t.exp(tr['psi'])
        mu = Infected
        p = r/(r+mu) + 1e-8


        if not self.proposal:
            tr('obs', alan.NegativeBinomial(total_count=r, logits=p))

class RandomWalkMobilityModel_ML(alan.AlanModule):
    def __init__(self, nRs, nWs, nCMs):
        super().__init__()
        self.CM_Alpha = alan.MLNormal(sample_shape=(nCMs-2,))

        self.Wearing_Alpha = alan.MLNormal()

        self.Mobility_Alpha = alan.MLNormal()

        self.RegionR = alan.MLNormal({'plate_nRs': nRs})

        self.GI_mean = alan.MLNormal()

        self.GI_sd = alan.MLNormal()

        self.InitialSize_log = alan.MLNormal({'plate_nRs': nRs})

        self.Log_Infected = alan.MLNormal({'plate_nRs': nRs, 'nWs':nWs})

        self.psi = alan.MLNormal()

    def forward(self, tr,
                ActiveCMs_NPIs,
                ActiveCMs_wearing,
                ActiveCMs_mobility,):

        tr('CM_Alpha', self.CM_Alpha())
        tr('Wearing_Alpha', self.Wearing_Alpha())
        tr('Mobility_Alpha', self.Mobility_Alpha())
        tr('RegionR', self.RegionR())
        tr('GI_mean', self.GI_mean())
        tr('GI_sd', self.GI_sd())
        tr('InitialSize_log', self.InitialSize_log())
        tr('Log_Infected', self.Log_Infected())
        tr('psi', self.psi())


class RandomWalkMobilityModel_Q(alan.AlanModule):
    def __init__(self, nRs, nWs, nCMs):
        super().__init__()
        self.CM_Alpha_mean = nn.Parameter(tr.zeros((nCMs-2,)))
        self.log_CM_Alpha_sigma = nn.Parameter(tr.zeros((nCMs-2,)))

        self.Wearing_Alpha_mean = nn.Parameter(tr.zeros(()))
        self.log_Wearing_Alpha_sigma = nn.Parameter(tr.zeros(()))

        self.Mobility_Alpha_mean = nn.Parameter(tr.zeros(()))
        self.log_Mobility_Alpha_sigma = nn.Parameter(tr.zeros(()))

        self.RegionR_mean = nn.Parameter(tr.zeros((nRs,),names=('plate_nRs',)))
        self.log_RegionR_sigma = nn.Parameter(tr.zeros((nRs,),names=('plate_nRs',)))

        self.GI_mean_mean = nn.Parameter(tr.zeros(()))
        self.log_GI_mean_sigma = nn.Parameter(tr.zeros(()))

        self.GI_sd_mean = nn.Parameter(tr.zeros(()))
        self.log_GI_sd_sigma = nn.Parameter(tr.zeros(()))

        self.InitialSize_log_mean = nn.Parameter(tr.zeros((nRs,),names=('plate_nRs',)))
        self.log_InitialSize_log_sigma = nn.Parameter(tr.zeros((nRs,),names=('plate_nRs',)))

        self.Log_Infected_mean = nn.Parameter(tr.zeros((nRs,nWs),names=('plate_nRs','nWs')))
        self.log_Log_Infected_sigma = nn.Parameter(tr.zeros((nRs,nWs),names=('plate_nRs','nWs')))

        self.psi_mean = nn.Parameter(tr.zeros(()))
        self.log_psi_sigma = nn.Parameter(tr.zeros(()))

    def forward(self, tr,
                ActiveCMs_NPIs,
                ActiveCMs_wearing,
                ActiveCMs_mobility,):

        tr('CM_Alpha', alan.Normal(self.CM_Alpha_mean, self.log_CM_Alpha_sigma.exp()))
        tr('Wearing_Alpha', alan.Normal(self.Wearing_Alpha_mean, self.log_Wearing_Alpha_sigma.exp()))
        tr('Mobility_Alpha', alan.Normal(self.Mobility_Alpha_mean, self.log_Mobility_Alpha_sigma.exp()))
        tr('RegionR', alan.Normal(self.RegionR_mean, self.log_RegionR_sigma.exp()))
        tr('GI_mean', alan.Normal(self.GI_mean_mean, self.log_GI_mean_sigma.exp()))
        tr('GI_sd', alan.Normal(self.GI_sd_mean, self.log_GI_sd_sigma.exp()))
        tr('InitialSize_log', alan.Normal(self.InitialSize_log_mean, self.log_InitialSize_log_sigma.exp()))
        tr('Log_Infected', alan.Normal(self.Log_Infected_mean, self.log_Log_Infected_sigma.exp()))
        tr('psi', alan.Normal(self.psi_mean, self.log_psi_sigma.exp()))
