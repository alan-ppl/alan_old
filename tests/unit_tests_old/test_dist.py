import sys
sys.path.append('..')

import torch
import alan
import unittest

from alan.dist import *
from functorch.dim import dims
from torch.nn.functional import softmax
import random


def get_sample_asserter(target_shape, target_names):
    def inner(sample):
        dims = generic_dims(sample)
        assert  generic_order(sample, dims).shape == target_shape
        for i in range(len(target_names)):
            assert dims[i] in set(target_names)
    return inner

def get_log_prob_asserter(log_prob_shape, log_prob_names):
    def inner(lp):
        dims = generic_dims(lp)
        shape = generic_order(lp, dims).shape
        for i in range(len(log_prob_names)):
            assert shape[i] in set(log_prob_shape)
            assert dims[i] in set(log_prob_names)
    return inner

def test_normal():
    

class TestTorchdimDist(unittest.TestCase):
    def test_sample(self):
        def test_loc_scale_dist(loc, scale, target_shape, target_names,
                                log_prob_shape, log_prob_names, sample_dim=()):
            '''
            Use distributions from loc-scale family as test cases.
            '''
            for dist in [Normal, Gumbel, Laplace, LogNormal, Cauchy]:
                assert_sample = get_sample_asserter(target_shape, target_names)
                assert_log_prob = get_log_prob_asserter(log_prob_shape, log_prob_names)


                sample = dist(loc, scale).sample(reparam=True, sample_dims=sample_dim)
                lp = dist(loc, scale).log_prob(sample)
                assert_sample(sample)
                assert_log_prob(lp)

        def test_probs_non_categ_dist(probs, target_shape, target_names,
                                log_prob_shape, log_prob_names, sample_dim=()):
            '''
            Use distributions from probs family (excluding categorical)
            as test cases.
            '''
            for dist in [Bernoulli, ContinuousBernoulli, Geometric]:
                assert_sample = get_sample_asserter(target_shape, target_names)
                assert_log_prob = get_log_prob_asserter(log_prob_shape, log_prob_names)


                sample = dist(probs).sample(reparam=False, sample_dims=sample_dim)
                lp = dist(probs).log_prob(sample)
                assert_sample(sample)
                assert_log_prob(lp)

        def test_probs_categ_dist(probs, target_shape, target_names,
                                log_prob_shape, log_prob_names, sample_dim=()):
            '''
            Use categorical
            as test cases.
            '''
            for dist in [Categorical]:
                assert_sample = get_sample_asserter(target_shape, target_names)
                assert_log_prob = get_log_prob_asserter(log_prob_shape, log_prob_names)


                sample = dist(probs).sample(reparam=False, sample_dims=sample_dim)
                lp = dist(probs).log_prob(sample)
                assert_sample(sample)
                assert_log_prob(lp)

        def test_probs_total_count_dist(total_count, probs, target_shape, target_names,
                                log_prob_shape, log_prob_names, sample_dim=()):
            '''
            Use that have probs and total count parameters
            as test cases.
            '''
            for dist in [Binomial, NegativeBinomial, Multinomial]:
                assert_sample = get_sample_asserter(target_shape, target_names)
                assert_log_prob = get_log_prob_asserter(log_prob_shape, log_prob_names)


                sample = dist(total_count,probs).sample(reparam=False, sample_dims=sample_dim)
                lp = dist(total_count,probs).log_prob(sample)
                assert_sample(sample)
                assert_log_prob(lp)

        def test_concentration_dist(concentration0, concentration1, target_shape, target_names,
                                log_prob_shape, log_prob_names, sample_dim=()):
            '''
            Use distributions that have concentration0, concentration1
            as test cases.
            '''
            for dist in [Beta, Kumaraswamy]:
                assert_sample = get_sample_asserter(target_shape, target_names)
                assert_log_prob = get_log_prob_asserter(log_prob_shape, log_prob_names)


                sample = dist(concentration0, concentration1).sample(reparam=True, sample_dims=sample_dim)
                lp = dist(concentration0, concentration1).log_prob(sample)
                assert_sample(sample)
                assert_log_prob(lp)

        def test_rate_dist(rate, target_shape, target_names,
                                log_prob_shape, log_prob_names, sample_dim=()):
            '''
            Use distributions that have rate
            as test cases.
            '''
            for dist in [Exponential, Poisson]:
                assert_sample = get_sample_asserter(target_shape, target_names)
                assert_log_prob = get_log_prob_asserter(log_prob_shape, log_prob_names)


                sample = dist(rate).sample(reparam=False, sample_dims=sample_dim)
                lp = dist(rate).log_prob(sample)
                assert_sample(sample)
                assert_log_prob(lp)

        def test_single_df_dist(df, target_shape, target_names,
                                log_prob_shape, log_prob_names, sample_dim=()):
            '''
            Use distributions that have df
            as test cases.
            '''
            for dist in [Chi2]:
                assert_sample = get_sample_asserter(target_shape, target_names)
                assert_log_prob = get_log_prob_asserter(log_prob_shape, log_prob_names)


                sample = dist(df).sample(reparam=False, sample_dims=sample_dim)
                lp = dist(df).log_prob(sample)
                assert_sample(sample)
                assert_log_prob(lp)

        def test_df_dist(df1, df2, target_shape, target_names,
                                log_prob_shape, log_prob_names, sample_dim=()):
            '''
            Use distributions that have df1, df2
            as test cases.
            '''
            for dist in [FisherSnedecor]:
                assert_sample = get_sample_asserter(target_shape, target_names)
                assert_log_prob = get_log_prob_asserter(log_prob_shape, log_prob_names)


                sample = dist(df1,df2).sample(reparam=False, sample_dims=sample_dim)
                lp = dist(df1,df2).log_prob(sample)
                assert_sample(sample)
                assert_log_prob(lp)

        def test_single_concentration_dist(concentration, target_shape, target_names,
                                log_prob_shape, log_prob_names, sample_dim=()):
            '''
            Use distributions that have concentration
            as test cases.
            '''
            for dist in [Dirichlet]:
                assert_sample = get_sample_asserter(target_shape, target_names)
                assert_log_prob = get_log_prob_asserter(log_prob_shape, log_prob_names)


                sample = dist(concentration).sample(reparam=False, sample_dims=sample_dim)
                lp = dist(concentration).log_prob(sample)
                assert_sample(sample)
                assert_log_prob(lp)

        def test_concentration_rate_dist(concentration, rate, target_shape, target_names,
                                log_prob_shape, log_prob_names, sample_dim=()):
            '''
            Use distributions that have concentration and rate
            as test cases.
            '''
            for dist in [Gamma]:
                assert_sample = get_sample_asserter(target_shape, target_names)
                assert_log_prob = get_log_prob_asserter(log_prob_shape, log_prob_names)


                sample = dist(concentration,rate).sample(reparam=False, sample_dims=sample_dim)
                lp = dist(concentration,rate).log_prob(sample)
                assert_sample(sample)
                assert_log_prob(lp)

        def test_single_scale_dist(scale, target_shape, target_names,
                                log_prob_shape, log_prob_names, sample_dim=()):
            '''
            Use distributions that have scale
            as test cases.
            '''
            for dist in [HalfCauchy, HalfNormal]:
                assert_sample = get_sample_asserter(target_shape, target_names)
                assert_log_prob = get_log_prob_asserter(log_prob_shape, log_prob_names)


                sample = dist(scale).sample(reparam=False, sample_dims=sample_dim)
                lp = dist(scale).log_prob(sample)
                assert_sample(sample)
                assert_log_prob(lp)

        def test_dim_concentration_dist(dim, concentration, target_shape, target_names,
                                log_prob_shape, log_prob_names, sample_dim=()):
            '''
            Use distributions that have dim and concentration
            as test cases.
            '''
            for dist in [LKJCholesky]:
                assert_sample = get_sample_asserter(target_shape, target_names)
                assert_log_prob = get_log_prob_asserter(log_prob_shape, log_prob_names)

                sample = dist(dim,concentration).sample(reparam=False, sample_dims=sample_dim)
                lp = dist(dim,concentration).log_prob(sample)
                assert_sample(sample)
                assert_log_prob(lp)

        def test_LowRankMultivariateNormal_dist(loc, cov_factor, cov_diag, target_shape, target_names,
                                log_prob_shape, log_prob_names, sample_dim=()):
            '''
            Use distributions that have dim and concentration
            as test cases.
            '''
            for dist in [LowRankMultivariateNormal]:
                assert_sample = get_sample_asserter(target_shape, target_names)
                assert_log_prob = get_log_prob_asserter(log_prob_shape, log_prob_names)

                sample = dist(loc, cov_factor, cov_diag).sample(reparam=False, sample_dims=sample_dim)
                lp = dist(loc, cov_factor, cov_diag).log_prob(sample)
                assert_sample(sample)
                assert_log_prob(lp)

        def test_MultivariateNormal_dist(loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, target_shape, target_names,
                                log_prob_shape, log_prob_names, sample_dim=()):
            '''
            Use distributions that have dim and concentration
            as test cases.
            '''
            for dist in [MultivariateNormal]:
                assert_sample = get_sample_asserter(target_shape, target_names)
                assert_log_prob = get_log_prob_asserter(log_prob_shape, log_prob_names)

                sample = dist(loc, covariance_matrix, precision_matrix, scale_tril).sample(reparam=False, sample_dims=sample_dim)
                lp = dist(loc, covariance_matrix, precision_matrix, scale_tril).log_prob(sample)
                assert_sample(sample)
                assert_log_prob(lp)

        A_size=4
        B_size=5
        C_size=6
        Plate_1_size = 2
        Plate_2_size = 3
        K_a, K_b, K_c, Plate_1, Plate_2 = dims(5 , [A_size,B_size,C_size,Plate_1_size,Plate_2_size])
        # Loc-Scale
        test_loc_scale_dist(
            loc=torch.randn(4, 3), scale=torch.ones(4, 3),
            target_shape=(4, 3), target_names=[],
            log_prob_shape = (), log_prob_names = []
        )
        test_loc_scale_dist(
            loc=torch.randn(A_size, 3)[K_a],
            scale=torch.ones(A_size, 3)[K_a],
            target_shape=(A_size, 3), target_names=[K_a],
            log_prob_shape = (A_size,), log_prob_names = [K_a]
        )
        test_loc_scale_dist(
            loc=torch.randn(A_size, 3)[K_a],
            scale=torch.ones(B_size, 3)[K_b],
            target_shape=(A_size, B_size, 3), target_names=[K_a, K_b],
            log_prob_shape = (A_size, B_size), log_prob_names = [K_a, K_b]
        )
        test_loc_scale_dist(
            loc=torch.randn(A_size, 3)[K_a],
            scale=torch.ones(B_size, 3)[K_b],
            target_shape=(Plate_1_size, A_size, B_size, 3), target_names=[Plate_1, K_a, K_b],
            log_prob_shape = (A_size, B_size,Plate_1_size), log_prob_names = [K_a, K_b, Plate_1],
            sample_dim=(Plate_1,)
        )
        test_loc_scale_dist(
            loc=torch.randn(A_size, 3)[K_a],
            scale=torch.ones(B_size, 3)[K_b],
            target_shape=(Plate_1_size, Plate_2_size, A_size, B_size, 3), target_names=[Plate_1, Plate_2, K_a, K_b],
            log_prob_shape = (A_size, B_size,Plate_1_size, Plate_2_size), log_prob_names = [K_a, K_b, Plate_1, Plate_2],
            sample_dim=(Plate_1,Plate_2)
        )
        ## Probs non-categorical
        test_probs_non_categ_dist(
            probs=softmax(torch.randn(4, 3), dim=1),
            target_shape=(4, 3), target_names=[],
            log_prob_shape = (), log_prob_names = []
        )
        test_probs_non_categ_dist(
            probs=softmax(torch.randn(A_size, 3), dim=1)[K_a],
            target_shape=(A_size, 3), target_names=[K_a],
            log_prob_shape = (A_size,), log_prob_names = [K_a]
        )
        test_probs_non_categ_dist(
            probs=softmax(torch.randn(A_size, 3), dim=1)[K_a],
            target_shape=(Plate_1_size, A_size, 3), target_names=[Plate_1, K_a],
            log_prob_shape = (A_size,Plate_1_size), log_prob_names = [K_a, Plate_1],
            sample_dim=(Plate_1,)
        )
        test_probs_non_categ_dist(
            probs=softmax(torch.randn(A_size, 3), dim=1)[K_a],
            target_shape=(Plate_1_size, Plate_2_size, A_size, 3), target_names=[Plate_1, Plate_2, K_a],
            log_prob_shape = (A_size,Plate_1_size, Plate_2_size), log_prob_names = [K_a, Plate_1, Plate_2],
            sample_dim=(Plate_1,Plate_2)
        )
        ## Probs categorical
        test_probs_categ_dist(
            probs=softmax(torch.randn(4, 3), dim=1),
            target_shape=(4,), target_names=[],
            log_prob_shape = (), log_prob_names = []
        )
        test_probs_categ_dist(
            probs=softmax(torch.randn(A_size, 3), dim=1)[K_a],
            target_shape=(A_size,), target_names=[K_a],
            log_prob_shape = (A_size,), log_prob_names = [K_a]
        )
        test_probs_categ_dist(
            probs=softmax(torch.randn(A_size, 3), dim=1)[K_a],
            target_shape=(Plate_1_size, A_size), target_names=[Plate_1, K_a],
            log_prob_shape = (Plate_1_size,A_size), log_prob_names = [K_a,Plate_1],
            sample_dim=(Plate_1,)
        )
        test_probs_categ_dist(
            probs=softmax(torch.randn(A_size, 3), dim=1)[K_a],
            target_shape=(Plate_1_size,Plate_2_size, A_size), target_names=[Plate_1, Plate_2, K_a],
            log_prob_shape = (A_size,Plate_1_size, Plate_2_size), log_prob_names = [K_a,Plate_1, Plate_2],
            sample_dim=(Plate_1,Plate_2)
        )
        #Probs total count
        test_probs_total_count_dist(
            total_count = random.randint(1,20),
            probs=softmax(torch.randn(4, 3), dim=1),
            target_shape=(4,3), target_names=[],
            log_prob_shape = (), log_prob_names = []
        )
        test_probs_total_count_dist(
            total_count = random.randint(1,20),
            probs=softmax(torch.randn(A_size, 3), dim=1)[K_a],
            target_shape=(A_size,3), target_names=[K_a],
            log_prob_shape = (A_size,), log_prob_names = [K_a]
        )
        test_probs_total_count_dist(
            total_count = random.randint(1,20),
            probs=softmax(torch.randn(A_size, 3), dim=1)[K_a],
            target_shape=(Plate_1_size, A_size, 3), target_names=[Plate_1, K_a],
            log_prob_shape = (A_size,Plate_1_size), log_prob_names = [K_a,Plate_1],
            sample_dim=(Plate_1,)
        )
        test_probs_total_count_dist(
            total_count = random.randint(1,20),
            probs=softmax(torch.randn(A_size, 3), dim=1)[K_a],
            target_shape=(Plate_1_size,Plate_2_size, A_size, 3), target_names=[Plate_1,Plate_2, K_a],
            log_prob_shape = (A_size,Plate_1_size,Plate_2_size), log_prob_names = [K_a,Plate_1,Plate_2],
            sample_dim=(Plate_1,Plate_2)
        )
        #Conentration
        test_concentration_dist(
            concentration0=torch.rand(4, 3), concentration1=torch.rand(4, 3),
            target_shape=(4, 3), target_names=[],
            log_prob_shape = (), log_prob_names = []
        )
        test_concentration_dist(
            concentration0=torch.rand(A_size, 3)[K_a], concentration1=torch.rand(A_size, 3)[K_a],
            target_shape=(A_size, 3), target_names=[K_a],
            log_prob_shape = (A_size,), log_prob_names = [K_a]
        )
        test_concentration_dist(
            concentration0=torch.rand(A_size, 3)[K_a], concentration1=torch.rand(B_size, 3)[K_b],
            target_shape=(A_size, B_size, 3), target_names=[K_a, K_b],
            log_prob_shape = (A_size, B_size), log_prob_names = [K_a, K_b]
        )
        test_concentration_dist(
            concentration0=torch.rand(A_size, 3)[K_a], concentration1=torch.rand(B_size, 3)[K_b],
            target_shape=(Plate_1_size, A_size, B_size, 3), target_names=[Plate_1, K_a, K_b],
            log_prob_shape = (A_size, B_size,Plate_1_size), log_prob_names = [K_a, K_b, Plate_1],
            sample_dim=(Plate_1,)
        )
        test_concentration_dist(
            concentration0=torch.rand(A_size, 3)[K_a], concentration1=torch.rand(B_size, 3)[K_b],
            target_shape=(Plate_1_size, Plate_2_size, A_size, B_size, 3), target_names=[Plate_1, Plate_2, K_a, K_b],
            log_prob_shape = (A_size, B_size,Plate_1_size,Plate_2_size), log_prob_names = [K_a, K_b, Plate_1, Plate_2],
            sample_dim=(Plate_1,Plate_2)
        )
        #single df
        test_single_df_dist(
            df=torch.randn(4, 3).exp(),
            target_shape=(4,3), target_names=[],
            log_prob_shape = (), log_prob_names = []
        )
        test_single_df_dist(
            df=torch.randn(4, 3).exp()[K_a],
            target_shape=(A_size,3), target_names=[K_a],
            log_prob_shape = (A_size,), log_prob_names = [K_a]
        )
        test_single_df_dist(
            df=torch.randn(4, 3).exp()[K_a],
            target_shape=(Plate_1_size, A_size, 3), target_names=[Plate_1, K_a],
            log_prob_shape = (A_size,Plate_1_size), log_prob_names = [K_a,Plate_1],
            sample_dim=(Plate_1,)
        )
        test_single_df_dist(
            df=torch.randn(4, 3).exp()[K_a],
            target_shape=(Plate_1_size,Plate_2_size, A_size, 3), target_names=[Plate_1,Plate_2, K_a],
            log_prob_shape = (A_size,Plate_1_size,Plate_2_size), log_prob_names = [K_a,Plate_1,Plate_2],
            sample_dim=(Plate_1,Plate_2)
        )
        #Rate
        test_rate_dist(
            rate=torch.randn(4, 3).exp(),
            target_shape=(4,3), target_names=[],
            log_prob_shape = (), log_prob_names = []
        )
        test_rate_dist(
            rate=torch.randn(4, 3).exp()[K_a],
            target_shape=(A_size,3), target_names=[K_a],
            log_prob_shape = (A_size,), log_prob_names = [K_a]
        )
        test_rate_dist(
            rate=torch.randn(4, 3).exp()[K_a],
            target_shape=(Plate_1_size, A_size, 3), target_names=[Plate_1, K_a],
            log_prob_shape = (A_size,Plate_1_size), log_prob_names = [K_a,Plate_1],
            sample_dim=(Plate_1,)
        )
        test_rate_dist(
            rate=torch.randn(4, 3).exp()[K_a],
            target_shape=(Plate_1_size,Plate_2_size, A_size, 3), target_names=[Plate_1,Plate_2, K_a],
            log_prob_shape = (A_size,Plate_1_size,Plate_2_size), log_prob_names = [K_a,Plate_1,Plate_2],
            sample_dim=(Plate_1,Plate_2)
        )
        #concentration
        test_single_concentration_dist(
            concentration=torch.randn(4, 3).exp(),
            target_shape=(4,3), target_names=[],
            log_prob_shape = (), log_prob_names = []
        )
        test_single_concentration_dist(
            concentration=torch.randn(4, 3).exp()[K_a],
            target_shape=(A_size,3), target_names=[K_a],
            log_prob_shape = (A_size,), log_prob_names = [K_a]
        )
        test_single_concentration_dist(
            concentration=torch.randn(4, 3).exp()[K_a],
            target_shape=(Plate_1_size, A_size, 3), target_names=[Plate_1, K_a],
            log_prob_shape = (A_size,Plate_1_size), log_prob_names = [K_a,Plate_1],
            sample_dim=(Plate_1,)
        )
        test_single_concentration_dist(
            concentration=torch.randn(4, 3).exp()[K_a],
            target_shape=(Plate_1_size,Plate_2_size, A_size, 3), target_names=[Plate_1,Plate_2, K_a],
            log_prob_shape = (A_size,Plate_1_size,Plate_2_size), log_prob_names = [K_a,Plate_1,Plate_2],
            sample_dim=(Plate_1,Plate_2)
        )
        #Df
        test_df_dist(
            df1=torch.rand(4, 3), df2=torch.rand(4, 3),
            target_shape=(4, 3), target_names=[],
            log_prob_shape = (), log_prob_names = []
        )
        test_df_dist(
            df1=torch.rand(A_size, 3)[K_a], df2=torch.rand(A_size, 3)[K_a],
            target_shape=(A_size, 3), target_names=[K_a],
            log_prob_shape = (A_size,), log_prob_names = [K_a]
        )
        test_df_dist(
            df1=torch.rand(A_size, 3)[K_a], df2=torch.rand(B_size, 3)[K_b],
            target_shape=(A_size, B_size, 3), target_names=[K_a, K_b],
            log_prob_shape = (A_size, B_size), log_prob_names = [K_a, K_b]
        )
        test_df_dist(
            df1=torch.rand(A_size, 3)[K_a], df2=torch.rand(B_size, 3)[K_b],
            target_shape=(Plate_1_size, A_size, B_size, 3), target_names=[Plate_1, K_a, K_b],
            log_prob_shape = (A_size, B_size,Plate_1_size), log_prob_names = [K_a, K_b, Plate_1],
            sample_dim=(Plate_1,)
        )
        test_df_dist(
            df1=torch.rand(A_size, 3)[K_a], df2=torch.rand(B_size, 3)[K_b],
            target_shape=(Plate_1_size, Plate_2_size, A_size, B_size, 3), target_names=[Plate_1, Plate_2, K_a, K_b],
            log_prob_shape = (A_size, B_size,Plate_1_size,Plate_2_size), log_prob_names = [K_a, K_b, Plate_1, Plate_2],
            sample_dim=(Plate_1,Plate_2)
        )
        #Conentration and rate
        test_concentration_rate_dist(
            concentration=torch.rand(4, 3), rate=torch.rand(4, 3),
            target_shape=(4, 3), target_names=[],
            log_prob_shape = (), log_prob_names = []
        )
        test_concentration_rate_dist(
            concentration=torch.rand(A_size, 3)[K_a], rate=torch.rand(A_size, 3)[K_a],
            target_shape=(A_size, 3), target_names=[K_a],
            log_prob_shape = (A_size,), log_prob_names = [K_a]
        )
        test_concentration_rate_dist(
            concentration=torch.rand(A_size, 3)[K_a], rate=torch.rand(B_size, 3)[K_b],
            target_shape=(A_size, B_size, 3), target_names=[K_a, K_b],
            log_prob_shape = (A_size, B_size), log_prob_names = [K_a, K_b]
        )
        test_concentration_rate_dist(
            concentration=torch.rand(A_size, 3)[K_a], rate=torch.rand(B_size, 3)[K_b],
            target_shape=(Plate_1_size, A_size, B_size, 3), target_names=[Plate_1, K_a, K_b],
            log_prob_shape = (A_size, B_size,Plate_1_size), log_prob_names = [K_a, K_b, Plate_1],
            sample_dim=(Plate_1,)
        )
        test_concentration_rate_dist(
            concentration=torch.rand(A_size, 3)[K_a], rate=torch.rand(B_size, 3)[K_b],
            target_shape=(Plate_1_size, Plate_2_size, A_size, B_size, 3), target_names=[Plate_1, Plate_2, K_a, K_b],
            log_prob_shape = (A_size, B_size,Plate_1_size,Plate_2_size), log_prob_names = [K_a, K_b, Plate_1, Plate_2],
            sample_dim=(Plate_1,Plate_2)
        )

        #scale
        test_single_scale_dist(
            scale=torch.randn(4, 3).exp(),
            target_shape=(4,3), target_names=[],
            log_prob_shape = (), log_prob_names = []
        )
        test_single_scale_dist(
            scale=torch.randn(4, 3).exp()[K_a],
            target_shape=(A_size,3), target_names=[K_a],
            log_prob_shape = (A_size,), log_prob_names = [K_a]
        )
        test_single_scale_dist(
            scale=torch.randn(4, 3).exp()[K_a],
            target_shape=(Plate_1_size, A_size, 3), target_names=[Plate_1, K_a],
            log_prob_shape = (A_size,Plate_1_size), log_prob_names = [K_a,Plate_1],
            sample_dim=(Plate_1,)
        )
        test_single_scale_dist(
            scale=torch.randn(4, 3).exp()[K_a],
            target_shape=(Plate_1_size,Plate_2_size, A_size, 3), target_names=[Plate_1,Plate_2, K_a],
            log_prob_shape = (A_size,Plate_1_size,Plate_2_size), log_prob_names = [K_a,Plate_1,Plate_2],
            sample_dim=(Plate_1,Plate_2)
        )
        #Dim concentration dists
        dim = random.randint(2,20)
        test_dim_concentration_dist(
            dim = dim,
            concentration=softmax(torch.randn(4, 3), dim=1),
            target_shape=(4,3,dim,dim), target_names=[],
            log_prob_shape = (), log_prob_names = []
        )
        test_dim_concentration_dist(
            dim = dim,
            concentration=softmax(torch.randn(A_size, 3), dim=1)[K_a],
            target_shape=(A_size,3, dim, dim), target_names=[K_a],
            log_prob_shape = (A_size,), log_prob_names = [K_a]
        )
        test_dim_concentration_dist(
            dim = dim,
            concentration=softmax(torch.randn(A_size, 3), dim=1)[K_a],
            target_shape=(Plate_1_size, A_size, 3, dim, dim), target_names=[Plate_1, K_a],
            log_prob_shape = (A_size,Plate_1_size), log_prob_names = [K_a,Plate_1],
            sample_dim=(Plate_1,)
        )
        test_dim_concentration_dist(
            dim = dim,
            concentration=softmax(torch.randn(A_size, 3), dim=1)[K_a],
            target_shape=(Plate_1_size,Plate_2_size, A_size, 3, dim, dim), target_names=[Plate_1,Plate_2, K_a],
            log_prob_shape = (A_size,Plate_1_size,Plate_2_size), log_prob_names = [K_a,Plate_1,Plate_2],
            sample_dim=(Plate_1,Plate_2)
        )
        #LowRankMultivariateNormal
        test_LowRankMultivariateNormal_dist(
            loc=torch.randn(4,3), cov_factor=torch.rand(4,3,2),
            cov_diag=torch.rand(4,3),
            target_shape=(4, 3), target_names=[],
            log_prob_shape = (), log_prob_names = []
        )
        test_LowRankMultivariateNormal_dist(
            loc=torch.randn(A_size, 3)[K_a], cov_factor=torch.rand(A_size, 3,2)[K_a],
            cov_diag=torch.rand(A_size, 3)[K_a],
            target_shape=(A_size, 3), target_names=[K_a],
            log_prob_shape = (A_size,), log_prob_names = [K_a]
        )
        test_LowRankMultivariateNormal_dist(
            loc=torch.randn(A_size, 3)[K_a], cov_factor=torch.rand(B_size, 3,2)[K_b],
            cov_diag=torch.rand(B_size, 3)[K_b],
            target_shape=(A_size, B_size, 3), target_names=[K_a, K_b],
            log_prob_shape = (A_size, B_size), log_prob_names = [K_a, K_b]
        )
        test_LowRankMultivariateNormal_dist(
            loc=torch.randn(A_size, 3)[K_a], cov_factor=torch.rand(B_size, 3,2)[K_b],
            cov_diag=torch.rand(B_size, 3)[K_b],
            target_shape=(Plate_1_size, A_size, B_size, 3), target_names=[Plate_1, K_a, K_b],
            log_prob_shape = (A_size, B_size,Plate_1_size), log_prob_names = [K_a, K_b, Plate_1],
            sample_dim=(Plate_1,)
        )
        test_LowRankMultivariateNormal_dist(
            loc=torch.randn(A_size, 3)[K_a], cov_factor=torch.rand(B_size, 3,2)[K_b],
            cov_diag=torch.rand(B_size, 3)[K_b],
            target_shape=(Plate_1_size, Plate_2_size, A_size, B_size, 3), target_names=[Plate_1, Plate_2, K_a, K_b],
            log_prob_shape = (A_size, B_size,Plate_1_size, Plate_2_size), log_prob_names = [K_a, K_b, Plate_1, Plate_2],
            sample_dim=(Plate_1,Plate_2)
        )



if __name__ == '__main__':
    unittest.main()
