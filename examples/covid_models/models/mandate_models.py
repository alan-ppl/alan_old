"""
Original Authors: Gavin Leech and Charlie Rogers Smith
Source: https://raw.githubusercontent.com/g-leech/masks_v_mandates/main/epimodel/pymc3_models/mandate_models.py
"""

# from pymc3 import Model
import torch as t
import torch.nn as nn
import alan
import numpy as np
# import pymc3 as pm
# import theano.tensor as T
# import theano.tensor.signal.conv as C

# All cases-only for now.

class HardcodedMandateModel(nn.Module):

    def __init__(self, df, nRs, delay=True, name='', model=None) :
        super().__init__()

        self.df = df
        self.nRs = nRs
        self.regions = list(df.index.get_level_values("city").unique())
        self.response = 'cases_cumulative'

        if delay:
            # Column with constant time shift
            self.feature = 'mandate_delayed'
        else :
            self.feature = 'mandate_effected'

        def forward(self, tr):
            intercept_mean_mean=20
            intercept_mean_sd=10
            intercept_var_sd=10

            ############
            # Intercept
            # Group mean
            tr.sample('a_grp', alan.Normal(intercept_mean_mean, intercept_mean_sd))
            # Group variance
            tr.sample('a_grp_sigma', alan.HalfNormal(intercept_var_sd))
            # Individual intercepts
            tr.sample('a_ind', alan.Normal(
                              mu=a_grp, sigma=a_grp_sigma)
                              plates=nRs)

            # Group mean
            tr.sample('b_grp', alan.Normal(1.33, .5))
            # Group variance
            tr.sample('b_grp_sigma', alan.HalfNormal(.5))
            # Individual slopes
            tr.sample('b_ind', alan.Normal('b_ind',
                              mu=b_grp, sigma=b_grp_sigma)
                              plates=nRs)

            # Group mean
            tr.sample('c_grp', alan.Normal(0, .5))
            # Group variance
            tr.sample('c_grp_sigma', alan.HalfNormal(.5))
            # Individual slopes
            tr.sample('c_ind', alan.Normal(
                              mu=c_grp, sigma=c_grp_sigma)
                              plates=nRs)

            # Error
            tr.sample('sigma', alan.HalfNormal(50.) plates=nRs)

            # Create likelihood for each city

            #We don't need this? just do normal tr.sample('obs') thing?
            for i, city in enumerate(self.regions):
                df_city = self.df.iloc[self.df.index.get_level_values('city') == city]

                # By using pm.Data we can change these values after sampling.
                # This allows us to extend x into the future so we can get
                # forecasts by sampling from the posterior predictive

                ## What to do here? We need a alan.Data method...
                x = pm.Data(city + "x_data",
                            np.arange(len(df_city)))
                confirmed = pm.Data(city + "y_data", df_city[self.response].astype('float64').values)

                # Likelihood
                tr.sample('obs', pm.NegativeBinomial(
                    city,
                    (a_ind[i] * (b_ind[i] + c_ind[i] * df_city[self.feature]) ** x), # Exponential regression
                    sigma[i],
                    observed=confirmed))


class ConfirmationDelayMandateModel(nn.Module):

    def __init__(self, df, nRs, delay=True, name='', model=None) :
        super().__init__()

        self.df = df
        self.nRs = nRs
        self.regions = list(df.index.get_level_values("city").unique())
        self.response = 'cases_cumulative'

        self.cases_delay_mean_mean = 10
        self.cases_delay_mean_sd = 1
        self.cases_truncation = 32


        def forward(self, tr):
            # Intercept
            # Group mean
            tr.sample('a_grp', alan.Normal(2, 10))
            # Group variance
            tr.sample('a_grp_sigma', alan.HalfNormal(10))
            # Individual intercepts
            tr.sample('a_ind', alan.Normal(
                              mu=a_grp, sigma=a_grp_sigma)
                              plates=nRs)

            # Group mean
            tr.sample('b_grp', alan.Normal(1.13, .5))
            # Group variance
            tr.sample('b_grp_sigma', alan.HalfNormal(.5))
            # Individual slopes
            tr.sample('b_ind', alan.Normal(mu=b_grp, sigma=b_grp_sigma)
                              plates=nRs)

            # Group mean
            tr.sample('c_grp', alan.Normal(0, .5))
            # Group variance
            tr.sample('c_grp_sigma', alan.HalfNormal(.5))
            # Individual slopes
            tr.sample('c_ind', alan.Normal(mu=c_grp, sigma=c_grp_sigma)
                              plates=nRs)
            # Error
            tr.sample('sigma', alan.HalfNormal(50.) plates=nRs)

            tr.sample('cases_delay_dist', alan.NegativeBinomial(mu=self.cases_delay_mean_mean, alpha=5))
            reporting_delay = self.truncate_and_normalise(tr['cases_delay_dist'])

            # Create likelihood for each city
            for i, city in enumerate(self.regions):
                df_city = df.iloc[df.index.get_level_values('city') == city]

                x = pm.Data(city + "x_data",
                            np.arange(len(df_city)))
                # Exponential regression
                infected_cases = (a_ind[i] * (b_ind[i] + c_ind[i] * df_city['mandate_effected']) ** x)
                infected_cases = T.reshape(infected_cases, (1, len(df_city)))

                # convolve with delay to produce expectations
                expected_cases = t.conv2d(
                  infected_cases,
                  reporting_delay,
                  border_mode="full"
                )[0, :len(df_city)]

                # By using pm.Data we can change these values after sampling.
                # This allows us to extend x into the future so we can get
                # forecasts by sampling from the posterior predictive
                confirmed = pm.Data(city + "y_data", df_city[self.response].astype('float64').values)

                # Likelihood
                pm.NegativeBinomial(
                    city,
                    mu=expected_cases,
                    alpha=sigma[i],
                    shape=len(df_city),
                    observed=confirmed)


    def truncate_and_normalise(self, delay) :
        bins = np.arange(0, self.cases_truncation)
        pmf = T.exp(delay.logp(bins))
        pmf = pmf / T.sum(pmf)

        return pmf.reshape((1, self.cases_truncation))
