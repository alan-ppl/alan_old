import sys
sys.path.append('..')
import torch
import torch.nn as nn
import tpp
from tpp.backend import vi

from torchdim import dims
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import unittest

# Testing
class tests(unittest.TestCase):
    def test_denaming(self):
        K_a, K_b= dims(2 , [4,4])
        sample = torch.randn(4, 4, 3)[K_a,K_b]

        dedimmed_sample = tpp.dename(sample)

        assert getattr(dedimmed_sample, 'dims', None) == None

    def test_get_dims(self):
        K_a, K_b= dims(2 , [4,4])
        sample = torch.randn(4, 4, 3)[K_a,K_b]


        assert tpp.tensor_utils.get_dims(sample)[0] is K_a
        assert tpp.tensor_utils.get_dims(sample)[1] is K_b

    def test_namify(self):
        K_a, K_b= dims(2 , [4,4])
        sample = torch.randn(4, 4, 3)[K_a,K_b]

        named_sample, _, denamify = tpp.nameify([sample])

        assert getattr(named_sample[0], 'dims', None) == None
        assert named_sample[0].names == ('K_a', 'K_b', None)

    def test_denamify(self):
        K_a, K_b= dims(2 , [4,4])
        sample = torch.randn(4, 4, 3)[K_a,K_b]

        named_sample, _, denamify = tpp.nameify([sample])

        dimmed_sample = denamify(named_sample[0])

        assert getattr(dimmed_sample, 'dims', None) == (K_a, K_b)
        assert dimmed_sample.names == (None,)



if __name__ == '__main__':
    unittest.main()
