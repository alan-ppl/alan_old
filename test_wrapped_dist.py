from os import register_at_fork
import torch
from tpp.wrapped_distribution import Normal, Gumbel, Laplace
import unittest


def get_asserter(target_shape, target_names):
    def inner(sample):
        assert sample.shape == target_shape
        assert sample.names == target_names
    return inner


class TestWrappedDist(unittest.TestCase):
    def test_sample(self):
        def test_loc_scale_dist(loc, scale, target_shape, target_names,
                                sample_shape=(), sample_names=()):
            '''
            Use distributions from loc-scale family as test cases.
            '''
            for dist in [Normal, Gumbel, Laplace]:
                assert_sample = get_asserter(target_shape, target_names)
                assert_sample(dist(loc, scale, sample_shape=sample_shape,
                                   sample_names=sample_names).rsample())
                assert_sample(dist(loc, scale=scale, sample_shape=sample_shape,
                                   sample_names=sample_names).rsample())
                assert_sample(dist(loc=loc, scale=scale, sample_shape=sample_shape,
                                   sample_names=sample_names).rsample())

        test_loc_scale_dist(
            loc=torch.randn(4, 3), scale=torch.ones(4, 3),
            target_shape=(4, 3), target_names=(None, None)
        )
        test_loc_scale_dist(
            loc=torch.randn(4, 3).refine_names('Ka', ...),
            scale=torch.ones(4, 3).refine_names('Ka', ...),
            target_shape=(4, 3), target_names=('Ka', None)
        )
        test_loc_scale_dist(
            loc=torch.randn(4, 3).refine_names('Ka', ...),
            scale=torch.ones(4, 3).refine_names('Kb', ...),
            target_shape=(4, 4, 3), target_names=('Ka', 'Kb', None)
        )
        test_loc_scale_dist(
            loc=torch.randn(4, 3).refine_names('Ka', ...),
            scale=torch.ones(4, 3).refine_names('Kb', ...),
            sample_shape=(3,), sample_names=('Plate1'),
            target_shape=(3, 4, 4, 3), target_names=('Plate1', 'Ka', 'Kb', None),
        )

    def test_log_prob(self):
        sample = torch.randn(4, 4, 3).refine_names('Ka', 'Kb', ...)
        dist = Normal(
            loc=torch.randn(4, 3).refine_names('Ka', ...),
            scale=torch.ones(4, 3).refine_names('Ka', ...)
        )
        ll = dist.log_prob(sample)
        assert ll.shape == (4, 4, 3)
        assert ll.names == ('Ka', 'Kb', None)

        sample = torch.randn(4, 4, 3).refine_names('Kb', 'Kc', ...)
        dist = Normal(
            loc=torch.randn(4, 4, 3).refine_names('Ka', 'Kc', ...),
            scale=torch.ones(4, 4, 3).refine_names('Ka', 'Kc', ...)
        )
        ll = dist.log_prob(sample)
        assert ll.shape == (4, 4, 4, 3)
        assert ll.names == ('Ka', 'Kb', 'Kc', None)


if __name__ == '__main__':
    unittest.main()
