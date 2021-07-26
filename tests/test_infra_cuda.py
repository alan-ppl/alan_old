import sys
sys.path.append('..')

import torch
import torch.nn.functional as F
from tpp.cartesian_tensor import CartesianTensor
import unittest


# Testing
class tests(unittest.TestCase):
    def test_cuda(self):
        '''
        Test a series of function that do not support broadcasting.
        '''
        # Test dot product
        a = CartesianTensor(torch.ones(5, 3).refine_names('Ka', ...).cuda())
        assert isinstance(a, CartesianTensor)
        assert str(a.device) == 'cuda:0'

        b = CartesianTensor(torch.ones(6, 3).refine_names('Kb', ...).cuda())
        assert isinstance(a + b, CartesianTensor)
        assert str((a + b).device) == 'cuda:0'


if __name__ == '__main__':
    if torch.cuda.is_available():
        unittest.main()
    else:
        print('No cuda found, exit')