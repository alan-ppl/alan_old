"""
Original Authors: Gavin Leech and Charlie Rogers Smith
Source: https://raw.githubusercontent.com/g-leech/masks_v_mandates/main/epimodel/pymc3_models/mask_models.py
"""

import torch as t
import torch.nn as nn
import alan
import numpy as np
import math

from .epi_params import EpidemiologicalParameters
# from epimodel.pymc3_distributions.asymmetric_laplace import AsymmetricLaplace

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
    def __init__(self,all_observed_active, nRs, nDs, nCMs, ActiveCMs, CMs, proposal=False):
        """
        Constructor.

        """
        super().__init__()
        self.nRs = t.tensor(nRs)
        self.nDs = t.tensor(nDs)
        self.nCMs = t.tensor(nCMs)
        self.all_observed_active = all_observed_active
        self.ActiveCMs = t.from_numpy(ActiveCMs)
        self.CMs = CMs


    def forward(
        self,
        tr,
        r_walk_period=7,
        r_walk_noise_scale_prior=0.15,
        intervention_prior="Gauss",
        cm_prior_scale=10,
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

        # build NPI Effectiveness priors
        if wearing_parameterisation is None:
            if intervention_prior == "AL":
                # tr('CM_Alpha', alan.AsymmetricLaplace(
                #     scale=cm_prior_scale,
                #     symmetry=0.5)
                #     plates='plate_CM_alpha',
                #     shape=(self.nCMs - 1,),
                # )
                pass
            else:
                pass
                # tr('CM_Alpha', alan.Normal(
                #     0, cm_prior_scale), plates='plate_CM_alpha', shape=(self.nCMs - 1,)
                # )
        else:
            assert self.CMs[-1] == "percent_mc"
            if intervention_prior == "AL":
                # tr('CM_Alpha', alan.AsymmetricLaplace(
                #     scale=cm_prior_scale,
                #     symmetry=0.5)
                #     plates='plate_CM_alpha',
                #     shape=(self.nCMs - 2,),
                # )
                pass
            else:
                # tr('CM_Alpha', alan.Normal(
                #     0, cm_prior_scale), plates='plate_CM_alpha',
                # )
                tr('CM_Alpha', alan.Normal(
                    t.zeros((self.nCMs -2,)), cm_prior_scale)
                )

        self.CMReduction = t.exp((-1.0) * tr['CM_Alpha'])

        # prior specification for wearing options:
        if wearing_parameterisation == "exp":
            tr("Wearing_Alpha", alan.Normal(wearing_mean, wearing_sigma), #shape=(1,)
            )
            self.WearingReduction = t.exp((-1.0) * tr['Wearing_Alpha'])
        # if wearing_parameterisation == "log_linear":
        #     tr("Wearing_Alpha", alan.Normal(wearing_mean_linear, wearing_sigma_linear), #shape=(1,)
        #     )
        #     self.WearingReduction = 1.0 - tr['Wearing_Alpha']
        # if wearing_parameterisation == "log_quadratic":
        #     tr("Wearing_Alpha", alan.Normal(wearing_mean_quadratic, wearing_sigma_quadratic), #shape=(1,)
        #     )
        #     self.WearingReduction = 1.0 - 2.0 * tr['Wearing_Alpha']
        # if wearing_parameterisation == "log_quadratic_2":
        #     tr("Wearing_Alpha", alan.Normal(wearing_mean_quadratic, wearing_sigma_quadratic), #shape=(2,)
        #     )
        #     self.WearingReduction = 1.0 - tr['Wearing_Alpha'][0] - tr['Wearing_Alpha'][1]
        tr('Mobility_Alpha', alan.Normal(
            mobility_mean, mobility_sigma), # shape=(1,)
        )
        self.MobilityReduction = (2.0 * (t.exp(-1.0 * tr['Mobility_Alpha']))) / (1.0 + t.exp(-1.0 * tr['Mobility_Alpha']))

        tr("HyperRMean", alan.TruncatedNormal(
            R_prior_mean_mean, R_prior_mean_scale, a=0.1
        ))

        tr("HyperRVar", alan.HalfNormal(R_noise_scale))

        # tr("RegionR_noise", alan.Normal(0, 1), plates='plate_nRs')# shape=(self.nRs,))
        tr("RegionR_noise", alan.Normal(t.zeros((self.nRs,)), 1))
        self.RegionR = tr['HyperRMean'] + tr['RegionR_noise'] * tr['HyperRVar']


        # load CMs active without wearing, compute log-R reduction and region log-R based on NPIs active
        if wearing_parameterisation is not None:
            #DATA
            self.ActiveCMs = self.ActiveCMs[:, :-2, :]
            # self.ActiveCMReduction = tr['CM_Alpha'] * self.ActiveCMs
            self.ActiveCMReduction = (
                                t.reshape(tr['CM_Alpha'], (1, self.nCMs - 2, 1)) * self.ActiveCMs
                            )

            self.ActiveCMs_wearing = self.ActiveCMs[:, -1, :]

        else:
            self.ActiveCMs = self.ActiveCMs[:, :-1, :]

            self.ActiveCMReduction = tr['CM_Alpha'] * self.ActiveCMs


        growth_reduction = t.sum(self.ActiveCMReduction, axis=1)

        if mob_and_wearing_only:
            growth_reduction = 0

        # calculating reductions for each of the wearing parameterisations
        if wearing_parameterisation == "exp":
            #DATA
            self.ActiveCMReduction_wearing = t.reshape(
                    tr['Wearing_Alpha'], (1, 1, 1)
                ) * t.reshape(
                    self.ActiveCMs_wearing,
                    (self.ActiveCMs.shape[0], 1, self.ActiveCMs.shape[2]),
                )
            growth_reduction_wearing = t.sum(self.ActiveCMReduction_wearing, axis=1)

        # if wearing_parameterisation == "log_linear":
        #     self.ActiveCMReduction_wearing = tr['Wearing_Alpha'] * self.ActiveCMs_wearing
        #     eps = 10 ** (-20)
        #     growth_reduction_wearing = -1.0 * t.log(
        #         t.nnet.relu(1.0 - t.sum(self.ActiveCMReduction_wearing, axis=1))
        #         + eps
        #     )
        #
        # if wearing_parameterisation == "log_quadratic":
        #     self.ActiveCMReduction_wearing = tr['Wearing_Alpha'] * self.ActiveCMs_wearing + t.reshape(tr['Wearing_Alpha'], (1, 1, 1))
        #         * self.ActiveCMs_wearing**2
        #
        #     eps = 10 ** (-20)
        #     growth_reduction_wearing = -1.0 * t.log(
        #         t.nnet.relu(1.0 - t.sum(self.ActiveCMReduction_wearing, axis=1))
        #         + eps
        #     )
        #
        # if wearing_parameterisation == "log_quadratic_2":
        #     self.ActiveCMReduction_wearing = (
        #         t.reshape(tr['Wearing_Alpha'][0], (1, 1, 1))
        #         * t.reshape(
        #             self.ActiveCMs_wearing,
        #             (self.ActiveCMs.shape[0], 1, self.ActiveCMs.shape[2]),
        #         )
        #         + t.reshape(tr['Wearing_Alpha'][1], (1, 1, 1))
        #         * t.reshape(
        #             self.ActiveCMs_wearing,
        #             (self.ActiveCMs.shape[0], 1, self.ActiveCMs.shape[2]),
        #         )
        #         ** 2
        #     )
        #     eps = 10 ** (-20)
        #     growth_reduction_wearing = -1.0 * t.log(
        #         t.nnet.relu(1.0 - t.sum(self.ActiveCMReduction_wearing, axis=1))
        #         + eps
        #     )
        if wearing_parameterisation is None:
            growth_reduction_wearing = 0
        else:
            sgrowth_reduction_wearing = growth_reduction_wearing

        # make reduction for mobility
        self.ActiveCMs_mobility =  self.ActiveCMs[:, -2, :]


        self.ActiveCMReduction_mobility = t.reshape(
                tr['Mobility_Alpha'], (1, 1, 1)
            ) * t.reshape(
                self.ActiveCMs_mobility,
                (self.ActiveCMs.shape[0], 1, self.ActiveCMs.shape[2]),
            )

        growth_reduction_mobility = -1.0 * t.log(
            t.sum(
                (2.0 * t.exp(-1.0 * self.ActiveCMReduction_mobility))
                / (1.0 + t.exp(-1.0 * self.ActiveCMReduction_mobility)),
                axis=1,
            )
        )

        if mobility_leaveout:
            growth_reduction_mobility = 0
            initial_mobility_reduction = 0
        else:
            initial_mobility_reduction = growth_reduction_mobility[:, 0]
            initial_mobility_reduction = t.reshape(initial_mobility_reduction, (self.nRs, 1))


        # random walk
        nNP = int(self.nDs / r_walk_period) - 1

        tr("r_walk_noise_scale", alan.HalfNormal(r_walk_noise_scale_prior))
        # rescaling variables by 10 for better NUTS adaptation
        # tr("r_walk_noise", alan.Normal(0, 1.0 / 10), plates=('plate_nRs','plate_nNP'))
        tr("r_walk_noise", alan.Normal(t.zeros((self.nRs,nNP)), 1.0 / 10))

        expanded_r_walk_noise = t.repeat_interleave(
            tr['r_walk_noise_scale'] * 10.0 * t.cumsum(tr['r_walk_noise'], axis=-1),
            r_walk_period,
            axis=-1,
        )[: self.nRs, : (self.nDs - 2 * r_walk_period)]

        full_log_Rt_noise = t.zeros((self.nRs, self.nDs))
        # full_log_Rt_noise = t.subtensor.set_subtensor(
        #     full_log_Rt_noise[:, 2 * r_walk_period :], expanded_r_walk_noise
        # )
        full_log_Rt_noise[:, 2 * r_walk_period :] = expanded_r_walk_noise

# ## ?????
#         transition = lambda x: alan.Normal(tr["r_walk_noise_scale"]*x, 0.1)
#         tr('r_walk_noise', alan.Timeseries(0, transition), T='plate_nNP')


        self.ExpectedLogR = t.reshape(t.log(self.RegionR), (self.nRs, 1)) \
            - growth_reduction \
            - growth_reduction_wearing \
            - (growth_reduction_mobility - initial_mobility_reduction) \
            + full_log_Rt_noise


        self.Rt_walk = t.exp(t.log(self.RegionR.reshape((self.nRs, 1))) + full_log_Rt_noise)


        self.Rt_cm = t.exp(
                t.log(self.RegionR.reshape((self.nRs, 1)))
                - growth_reduction
                - growth_reduction_wearing
            )


        # convert R into growth rates
        tr('GI_mean', alan.Normal(gi_mean_mean, gi_mean_sd))
        tr("GI_sd", alan.Normal(gi_sd_mean, gi_sd_sd))

        gi_beta = tr['GI_mean'] / tr['GI_sd'] ** 2
        gi_alpha = tr['GI_mean'] ** 2 / tr['GI_sd'] ** 2

        self.ExpectedGrowth = gi_beta * (
                np.exp(self.ExpectedLogR / gi_alpha)
                - t.ones_like(self.ExpectedLogR)
            )

        self.Growth = self.ExpectedGrowth

        # Originally N(0, 50)
        tr("InitialSize_log", alan.Normal(t.tensor(log_init_mean).repeat(self.nRs), log_init_sd))
        self.Infected_log = t.reshape(tr['InitialSize_log'], (self.nRs, 1)) \
            + self.Growth.cumsum(axis=1)

        self.Infected = t.exp(self.Infected_log)
        r = cases_delay_disp_mean
        mu = cases_delay_mean_mean
        p = r/(r+mu)

        # tr('cases_delay_dist', alan.NegativeBinomial(total_count=r, probs=p))

        cases_delay_dist = alan.NegativeBinomial(total_count=r, probs=p)
        bins = t.arange(0, cases_truncation)

        pmf = t.exp(cases_delay_dist.log_prob(bins))

        pmf = pmf / t.sum(pmf)
        reporting_delay = pmf.reshape((1, cases_truncation))

        ## Border=full?
        expected_confirmed = t.nn.functional.conv2d(
            self.Infected.float().reshape(1,1,self.Infected.shape[0],self.Infected.shape[1]),
            reporting_delay.reshape(1,1,reporting_delay.shape[0],reporting_delay.shape[1])
        )#[:, : self.nDs]
        print(expected_confirmed.shape)
        print(self.nRs)
        print(self.nDs)
        self.ExpectedCases = expected_confirmed.reshape((self.nRs, self.nDs))


        # Observation Noise Dispersion Parameter (negbin alpha)
        tr('Psi', alan.HalfNormal(5))
        # effectively handle missing values ourselves
        # likelihood
        r = self.Psi
        mu = self.ExpectedCases.reshape((self.nRs * self.nDs,))[
            self.all_observed_active
        ]
        p = r/(r+mu)
        # self.ObservedCases = pm.NegativeBinomial(
        #     "ObservedCases",
        #     self.ExpectedCases.reshape((self.nRs * self.nDs,))[
        #         self.all_observed_active
        #     ],
        #     alpha=self.Psi,
        #     shape=(len(self.all_observed_active),),
        #     observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
        #         self.all_observed_active
        #     ],
        # )
        if not proposal:
            tr('obs', alan.NegativeBinomial(total_count=r, probs=p), plates='Plate_obs',)

# class RandomWalkMobilityModel_Q(alan.QModule):
#     def __init__(self):

class MandateMobilityModel(nn.Module):
    def __init__(self,all_observed_active, nRs, nDs, nCMs, ActiveCMs, CMs, proposal=False):
        """
        Constructor.

        """
        super().__init__()
        self.nRs = nRs
        self.nDs = nDs
        self.nCMs = nCMs
        self.all_observed_active = all_observed_active
        self.ActiveCMs = ActiveCMs
        self.CMs = CMs

    def build_model(
        self,
        tr,
        r_walk_period=7,
        r_walk_noise_scale_prior=0.15,
        intervention_prior="Gauss",
        cm_prior_scale=10,
        mobility_mean=1.704,
        mobility_sigma=0.44,
        R_prior_mean_mean=1.07,
        R_prior_mean_scale=0.2,
        R_noise_scale=0.4,  # 0.5
        cm_prior="skewed",
        gi_mean_mean=5.06,
        gi_mean_sd=0.33,
        gi_sd_mean=2.11,
        gi_sd_sd=0.5,
        mask_sigma=0.08,
        n_mandates=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=10.92,
        cases_delay_mean_sd=2,
        cases_delay_disp_mean=5.41,
        cases_delay_disp_sd=1,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        mobility_leaveout=False,
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

        # mob_feature = "avg_mobility_no_parks_no_residential"
        # assert self.d.CMs[-2] == mob_feature
        # assert self.d.CMs[-1] == "H6_Facial Coverings"

        with self.model:
            # build NPI Effectiveness priors:
            if intervention_prior == "AL":
                # self.CM_Alpha = AsymmetricLaplace(
                #     "CM_Alpha",
                #     scale=cm_prior_scale,
                #     symmetry=0.5,
                #     shape=(self.nCMs - 3,),
                # )
                pass
            else:
                tr('CM_alpha', alan.normal(0, cm_prior_scale), shape=(self.nCMs - 3,))

            self.CMReduction = t.exp((-1.0) * tr['CM_Alpha'])



            tr('Mandate_Alpha_1', alan.Nmal(0, mask_sigma), shape=(1,))

            tr('Mandate_Alpha_2', alan.Normal(0, mask_sigma), shape=(1,))
            if n_mandates == 1:
                self.MandateReduction = t.exp((-1.0) * tr['Mandate_Alpha_1'])

            else:
                self.MandateReduction = t.exp((-1.0) * (tr['Mandate_Alpha_1'] + tr['Mandate_Alpha_2']))


            tr('Mobility_Alpha', alan.Normal(mobility_mean, mobility_sigma), shape=(1,))
            self.MobilityReduction = (2.0 * (t.exp(-1.0 * tr['Mobility_Alpha']))) / (1.0 + t.exp(-1.0 * tr['Mobility_Alpha']))


            # self.HyperRMean = pm.TruncatedNormal(
            #     "HyperRMean", R_prior_mean_mean, sigma=R_prior_mean_scale, lower=0.1
            # )
            tr('HyperRMean', alan.TruncatedNormal(R_prior_mean_mean, R_prior_mean_scale, 0.1))

            # self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=R_noise_scale)
            tr('HyperRVar', alan.HalfNormal(R_noise_scale))
            # self.RegionR_noise = pm.Normal("RegionR_noise", 0, 1, shape=(self.nRs,))
            tr('RegionR_noise', alan.Normal(0,1), plates='Plate_nRs')
            self.RegionR = self.HyperRMean + self.RegionR_noise * self.HyperRVar

            #DATA
            self.ActiveCMs = self.ActiveCMs[:, :-3, :]
            self.ActiveCMReduction = (
                t.reshape(tr['CM_Alpha'], (1, self.nCMs - 3, 1)) * self.ActiveCMs
            )

            self.ActiveCMs_mandate_1 = self.ActiveCMs[:, -3, :]

            self.ActiveCMs_mandate_2 = self.ActiveCMs[:, -1, :]


            growth_reduction = t.sum(self.ActiveCMReduction, axis=1)
            # pm.Deterministic("growth_reduction", growth_reduction)

            self.ActiveCMReduction_mandate_1 = t.reshape(
                tr['Mandate_Alpha_1'], (1, 1, 1)
            ) * t.reshape(
                self.ActiveCMs_mandate_1,
                (self.ActiveCMs.shape[0], 1, self.ActiveCMs.shape[2]),
            )
            self.ActiveCMReduction_mandate_2 = t.reshape(
                tr['Mandate_Alpha_2'], (1, 1, 1)
            ) * t.reshape(
                self.ActiveCMs_mandate_2,
                (self.ActiveCMs.shape[0], 1, self.ActiveCMs.shape[2]),
            )

            growth_reduction_mandate_1 = t.sum(self.ActiveCMReduction_mandate_1, axis=1)
            growth_reduction_mandate_2 = t.sum(self.ActiveCMReduction_mandate_2, axis=1)

            if n_mandates == 1:
                growth_reduction_mandate = growth_reduction_mandate_1
            else:
                growth_reduction_mandate = growth_reduction_mandate_1 + growth_reduction_mandate_2

            # make reduction for mobility
            #DATA
            self.ActiveCMs_mobility = self.ActiveCMs[:, -2, :]


            self.ActiveCMReduction_mobility = t.reshape(
                tr['Mobility_Alpha'], (1, 1, 1)
            ) * t.reshape(
                self.ActiveCMs_mobility,
                (self.ActiveCMs.shape[0], 1, self.ActiveCMs.shape[2]),
            )

            growth_reduction_mobility = -1.0 * t.log(
                t.sum(
                    (2.0 * t.exp(-1.0 * self.ActiveCMReduction_mobility))
                    / (1.0 + t.exp(-1.0 * self.ActiveCMReduction_mobility)),
                    axis=1,
                )
            )
            if mobility_leaveout:
                growth_reduction_mobility = 0
                initial_mobility_reduction = 0
            else:
                initial_mobility_reduction = growth_reduction_mobility[:, 0]
                initial_mobility_reduction = t.reshape(initial_mobility_reduction, (self.nRs, 1))
                # pm.Deterministic("initial_mobility_reduction", initial_mobility_reduction)
                #
                # pm.Deterministic("growth_reduction_mobility", growth_reduction_mobility)

            # random walk
            nNP = int(self.nDs / r_walk_period) - 1


            tr("r_walk_noise_scale", alan.HalfNormal(r_walk_noise_scale_prior))
            # rescaling variables by 10 for better NUTS adaptation
            # r_walk_noise = pm.Normal("r_walk_noise", 0, 1.0 / 10, shape=(self.nRs, nNP))
            tr("r_walk_noise", alan.Normal(0, 1.0 / 10), plates=('plate_nRs','plate_nNP'))

            expanded_r_walk_noise = t.repeat(
                r_walk_noise_scale * 10.0 * t.cumsum(r_walk_noise, axis=-1),
                r_walk_period,
                axis=-1,
            )[: self.nRs, : (self.nDs - 2 * r_walk_period)]

            full_log_Rt_noise = t.zeros((self.nRs, self.nDs))
            full_log_Rt_noise = t.subtensor.set_subtensor(
                full_log_Rt_noise[:, 2 * r_walk_period :], expanded_r_walk_noise
            )

            self.ExpectedLogR = t.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) \
                - growth_reduction \
                - growth_reduction_mandate \
                - (growth_reduction_mobility - initial_mobility_reduction) \
                + full_log_Rt_noise


            self.Rt_walk = t.exp(t.log(self.RegionR.reshape((self.nRs, 1))) + full_log_Rt_noise),


            self.Rt_cm = t.exp(
                    t.log(self.RegionR.reshape((self.nRs, 1)))
                    - growth_reduction
                    - growth_reduction_mandate
                )


            # convert R into growth rates
            tr("GI_mean", alan.Normal(gi_mean_mean, gi_mean_sd))

            tr("GI_sd", alan.Normal(gi_sd_mean, gi_sd_sd))

            gi_beta = tr['GI_mean'] / tr['GI_sd'] ** 2
            gi_alpha = tr['GI_mean'] ** 2 / tr['GI_sd'] ** 2

            self.ExpectedGrowth = gi_beta \
                * (
                    np.exp(self.ExpectedLogR / gi_alpha)
                    - t.ones_like(self.ExpectedLogR)
                )


            self.Growth = self.ExpectedGrowth

            # Originally N(0, 50)
            tr("InitialSize_log", alan.Normal(log_init_mean, log_init_sd), shape=(self.nRs,))
            self.Infected_log = t.reshape(tr["InitialSize_log"], (self.nRs, 1)) \
                + self.Growth.cumsum(axis=1)

            self.Infected = math.exp(self.Infected_log)

            r = cases_delay_disp_mean
            mu = cases_delay_mean_mean
            p = r/(r+mu)

            # tr('cases_delay_dist', alan.NegativeBinomial(total_count=r, probs=p))

            cases_delay_dist = alan.NegativeBinomial(total_count=r, probs=p)
            bins = np.arange(0, cases_truncation)
            pmf = t.exp(cases_delay_dist.logp(bins))
            pmf = pmf / t.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = t.nn.Functional.conv2d(
                self.Infected, reporting_delay)[:, : self.nDs]

            self.ExpectedCases = expected_confirmed.reshape((self.nRs, self.nDs))


            # Observation Noise Dispersion Parameter (negbin alpha)
            tr('Psi', alan.HalfNormal(5))

            # effectively handle missing values ourselves
            # likelihood
            # self.ObservedCases = pm.NegativeBinomial(
            #     "ObservedCases",
            #     self.ExpectedCases.reshape((self.nRs * self.nDs,))[
            #         self.all_observed_active
            #     ],
            #     alpha=self.Psi,
            #     shape=(len(self.all_observed_active),),
            #     observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
            #         self.all_observed_active
            #     ],
            # )
            r = self.Psi
            mu = self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                self.all_observed_active
            ]
            if not proposal:
                tr('obs', alan.NegativeBinomial(total_count=r, probs=p),shape=(len(self.all_observed_active),))
