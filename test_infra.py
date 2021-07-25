import torch
import torch.nn.functional as F
from tpp.cartesian_tensor import CartesianTensor
import unittest

# Testing
class tests(unittest.TestCase):
    def test_special_functions(self):
        '''
        Test a series of function that do not support broadcasting.
        '''
        # Test dot product
        a = CartesianTensor(torch.ones(5, 6, 3).refine_names('Ka', 'Kb', ...))
        b = CartesianTensor(torch.ones(6, 7, 3).refine_names('Kb', 'Kc', ...))
        out = torch.dot(a, b)
        assert out.shape == (5, 6, 7)
        assert out.names == ('Ka', 'Kb', 'Kc')
        assert type(out) == type(a)

        # Test outer product
        a = CartesianTensor(torch.ones(5, 6, 3).refine_names('Ka', 'Kb', ...))
        b = CartesianTensor(torch.ones(6, 7, 4).refine_names('Kb', 'Kc', ...))
        out = torch.outer(a, b)
        assert out.shape == (5, 6, 7, 3, 4)
        assert out.names == ('Ka', 'Kb', 'Kc', None, None)
        assert type(out) == type(a)

        # Test F.linear
        x = CartesianTensor(torch.ones(2, 3, 4, 10).refine_names('Ka', 'Kb', 'Kc', ...))
        W = torch.randn(5, 10)
        out = F.linear(x, W)
        assert out.shape == (2, 3, 4, 5)
        assert out.names == ('Ka', 'Kb', 'Kc', None)

        x = CartesianTensor(torch.ones(2, 3, 4, 10).refine_names('Ka', 'Kb', 'Kc', ...))
        W = CartesianTensor(torch.ones(3, 5, 5, 10).refine_names('Kb', 'Kd', ...))
        out = F.linear(x, W)
        assert out.shape == (2, 3, 4, 5, 5)
        assert out.names == ('Ka', 'Kb', 'Kc', 'Kd', None)

        ''' The following test cases skipped due to problem in F.linear:
            # https://github.com/pytorch/pytorch/issues/61544
        x = CartesianTensor(torch.ones(2, 3, 4, 10)).refine_names('Ka', 'Kb', 'Kc', ...)
        W = CartesianTensor(torch.ones(3, 5, 5, 10)).refine_names('Kb', 'Kd', ...)
        b = torch.randn(5,)
        out = F.linear(x, W, b)
        assert out.shape == (2, 3, 4, 5, 5)
        assert out.names == ('Ka', 'Kb', 'Kc', 'Kd', None)

        x = CartesianTensor(torch.ones(2, 3, 4, 10)).refine_names('Ka', 'Kb', 'Kc', ...)
        W = CartesianTensor(torch.ones(3, 5, 5, 10)).refine_names('Kb', 'Kd', ...)
        b = CartesianTensor(torch.ones(2, 4, 5)).refine_names('Ka', 'Kc', ...)
        out = F.linear(x, W, b)
        assert out.shape == (2, 3, 4, 5, 5)
        assert out.names == ('Ka', 'Kb', 'Kc', 'Kd', None)
        '''

        # Test F.conv2d
        x = CartesianTensor(torch.ones(2, 3, 4, 1, 8, 16, 16).refine_names('Ka', 'Kb', 'Kc', ...))
        W = CartesianTensor(torch.ones(3, 5, 16, 8, 5, 5).refine_names('Kb', 'Kd', ...))
        b = torch.randn(16,)
        out = F.conv2d(x, W, b, padding=2, stride=1)
        assert out.shape == (2, 3, 4, 5, 1, 16, 16, 16)
        assert out.names == ('Ka', 'Kb', 'Kc', 'Kd', *((None,)*4))

        print('Special function test passed...')


    def test_operators(self):
        operators = ['+', '-', '*', '/', '**']
        # Base cases
        a = CartesianTensor(torch.ones(6, 3).refine_names('Ka', ...))
        b = CartesianTensor(torch.ones(6, 3).refine_names('Kb', ...))
        for op in operators:
            out = eval(f'a {op} b')
            assert out.shape == (6, 6, 3)
            assert out.names == ('Ka', 'Kb', None)
            assert type(out) == type(a)

        a = CartesianTensor(torch.ones(6, 2, 3, 4).refine_names('Ka', ...))
        b = CartesianTensor(torch.ones(6, 4).refine_names('Kb', ...))
        for op in operators:
            out = eval(f'a {op} b')
            assert out.shape == (6, 6, 2, 3, 4)
            assert out.names == ('Ka', 'Kb', None, None, None)
            assert type(out) == type(a)

        a = CartesianTensor(torch.ones(6, 1, 1, 3).refine_names('Ka', ...))
        b = CartesianTensor(torch.ones(6, 3).refine_names('Kb', ...))
        for op in operators:
            out = eval(f'a {op} b')
            assert out.shape == (6, 6, 1, 1, 3)
            assert out.names == ('Ka', 'Kb', None, None, None)
            assert type(out) == type(a)

        # Share one common name
        a = CartesianTensor(torch.ones(5, 6, 3).refine_names('Ka', 'Kb', ...))
        b = CartesianTensor(torch.ones(6, 7, 3).refine_names('Kb', 'Kc', ...))
        for op in operators:
            out = eval(f'a {op} b')
            assert out.shape == (5, 6, 7, 3)
            assert out.names == ('Ka', 'Kb', 'Kc', None)
            assert type(out) == type(a)

        a = CartesianTensor(torch.ones(5, 6, 3).refine_names('Ka', 'Kb', ...))
        b = CartesianTensor(torch.ones(6, 7, 8, 3).refine_names(
            'Kb', 'Kc', 'Kd', ...))
        for op in operators:
            out = eval(f'a {op} b')
            assert out.shape == (5, 6, 7, 8, 3)
            assert out.names == ('Ka', 'Kb', 'Kc', 'Kd', None)
            assert type(out) == type(a)

        # Share multiple common names
        a = CartesianTensor(torch.ones(5, 6, 8, 3).refine_names(
            'Ka', 'Kb', 'Kd', ...))
        b = CartesianTensor(torch.ones(6, 7, 8, 3).refine_names(
            'Kb', 'Kc', 'Kd', ...))
        for op in operators:
            out = eval(f'a {op} b')
            assert out.shape == (5, 6, 7, 8, 3)
            assert out.names == ('Ka', 'Kb', 'Kc', 'Kd', None)
            assert type(out) == type(a)

        # Mixed CartesianTensor and normal tensor
        a = CartesianTensor(torch.ones(5, 6, 3).refine_names('Ka', 'Kb', ...))
        b = torch.randn(3)
        for op in operators:
            out = eval(f'a {op} b')
            assert out.shape == (5, 6, 3)
            assert out.names == ('Ka', 'Kb', None)
            assert type(out) == type(a)
            out = eval(f'b {op} a')
            assert out.shape == (5, 6, 3)
            assert out.names == ('Ka', 'Kb', None)
            assert type(out) == type(a)

        a = CartesianTensor(torch.ones(5, 6).refine_names('Ka', 'Kb'))
        b = torch.randn(3)
        for op in operators:
            out = eval(f'a {op} b')
            assert out.shape == (5, 6, 3)
            assert out.names == ('Ka', 'Kb', None)
            out = eval(f'b {op} a')
            assert out.shape == (5, 6, 3)
            assert out.names == ('Ka', 'Kb', None)
            assert type(out) == type(a)

        a = CartesianTensor(torch.ones(5, 6).refine_names('Ka', 'Kb'))
        b = torch.randn(3, 2, 3)
        for op in operators:
            out = eval(f'a {op} b')
            assert out.shape == (5, 6, 3, 2, 3)
            assert out.names == ('Ka', 'Kb', None, None, None)
            out = eval(f'b {op} a')
            assert out.shape == (5, 6, 3, 2, 3)
            assert out.names == ('Ka', 'Kb', None, None, None)
            assert type(out) == type(a)
        print("Operator test passed...")


    def test_chain_operations(self):
        k_list = ['Ka', 'Kb', 'Kc', 'Kd', 'Ke', 'Kf']
        base_udf_shape = (4,)
        k_shape = (5,)
        base_tensor = torch.randn(4,)  # Base udf tensor
        for k in k_list:
            base_tensor = base_tensor + \
                CartesianTensor(torch.ones(k_shape).refine_names(k,))
        assert base_tensor.shape == len(k_list) * k_shape + base_udf_shape
        assert base_tensor.names == tuple(k_list) + (None,)
        print("Chain operations test passed...")


    def test_func_call(self):
        '''
        Test different ways of calling functions
        '''
        a = CartesianTensor(torch.ones(5, 6).refine_names('Ka', 'Kb'))
        b = torch.randn(3, 2, 3)
        out = a.add(b)
        assert out.shape == (5, 6, 3, 2, 3)
        assert out.names == ('Ka', 'Kb', None, None, None)
        out = b.add(a)
        assert out.shape == (5, 6, 3, 2, 3)
        assert out.names == ('Ka', 'Kb', None, None, None)
        assert type(out) == type(a)
        out = torch.add(a, b)
        assert out.shape == (5, 6, 3, 2, 3)
        assert out.names == ('Ka', 'Kb', None, None, None)
        assert type(out) == type(a)
        print("Func call approaches test passed...")


    def test_reduction_op(self):
        a = CartesianTensor(torch.ones(5, 6, 3, 4).refine_names('Ka', 'Kb', ...))
        out = a.sum(0)
        assert out.shape == (5, 6, 4)
        out = a.sum(-1)
        assert out.shape == (5, 6, 3)
        out = a.sum(dim=-1)
        assert out.shape == (5, 6, 3)
        out = a.sum(dim=(0, 1))
        assert out.shape == (5, 6)
        print("Reduction Op test passed...")


if __name__ == '__main__':
    unittest.main()