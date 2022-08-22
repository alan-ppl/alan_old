import sys
sys.path.append('..')

import torch
import tpp
from tpp.wrapped_distribution import Normal, Gumbel, Laplace
import unittest

from functorch.dim import dims

def get_asserter(target_shape, target_names):
    def inner(sample):

        assert tpp.dename(sample).shape == target_shape

        for i in range(len(target_names)):
            assert tpp.tensor_utils.get_dims(sample)[i] is target_names[i]
    return inner


class TestWrappedDist(unittest.TestCase):
    def test_sample(self):
        def test_loc_scale_dist(loc, scale, target_shape, target_names,
                                sample_dim=()):
            '''
            Use distributions from loc-scale family as test cases.
            '''
            for dist in [Normal, Gumbel, Laplace]:
                assert_sample = get_asserter(target_shape, target_names)
                for i in range(3):
                    assert_sample(dist(loc, scale, sample_dim=sample_dim).rsample())

        K_a, K_b, K_c, Plate_1 = dims(4 , [4,4,4,3])

        test_loc_scale_dist(
            loc=torch.randn(4, 3), scale=torch.ones(4, 3),
            target_shape=(4, 3), target_names=[]
        )
        test_loc_scale_dist(
            loc=torch.randn(4, 3)[K_a],
            scale=torch.ones(4, 3)[K_a],
            target_shape=(4, 3), target_names=[K_a]
        )
        test_loc_scale_dist(
            loc=torch.randn(4, 3)[K_a],
            scale=torch.ones(4, 3)[K_b],
            target_shape=(4, 4, 3), target_names=[K_a, K_b]
        )
        test_loc_scale_dist(
            loc=torch.randn(4, 3)[K_a],
            scale=torch.ones(4, 3)[K_b],
            sample_dim=Plate_1,
            target_shape=(3, 4, 4, 3), target_names=[Plate_1, K_a, K_b],
        )

    def test_log_prob(self):
        K_a, K_b, K_c, Plate_1 = dims(4 , [4,4,4,3])
        sample = torch.randn(4, 4, 3)[K_a,K_b]
        dist = Normal(
            loc=torch.randn(4, 3)[K_a],
            scale=torch.ones(4, 3)[K_b]
        )
        ll = dist.log_prob(sample)
        assert ll.shape == (4, 4, 3)
        assert ll.names == ('K_a', 'K_b', None)

        sample = torch.randn(4, 4, 3)[K_b, K_c]
        dist = Normal(
            loc=torch.randn(4, 4, 3)[K_a, K_c],
            scale=torch.ones(4, 4, 3)[K_a, K_c]
        )
        ll = dist.log_prob(sample)
        assert ll.shape == (4, 4, 4, 3)
        assert ll.names == ('K_a', 'K_b', 'K_c', None)


if __name__ == '__main__':
    unittest.main()
