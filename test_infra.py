import torch
from attic.cartesian_tensor import CartesianTensor, _process_common_names, _get_name
from functools import partial

# Testing

def test_special_functions():
    '''
    Test a series of function that do not support broadcasting.
    '''
    # Test dot product
    a = CartesianTensor(torch.ones(5, 6, 3)).refine_names('Ka','Kb', ...)
    b = CartesianTensor(torch.ones(6, 7, 3)).refine_names('Kb','Kc', ...)
    out = torch.dot(a, b)
    assert out.shape == (5, 6, 7)
    assert out.names == ('Ka', 'Kb', 'Kc')
    assert type(out) == type(a)

    # Test outer product
    a = CartesianTensor(torch.ones(5, 6, 3)).refine_names('Ka','Kb', ...)
    b = CartesianTensor(torch.ones(6, 7, 4)).refine_names('Kb','Kc', ...)
    out = torch.outer(a, b)
    assert out.shape == (5, 6, 7, 3, 4)
    assert out.names == ('Ka', 'Kb', 'Kc', None, None)
    assert type(out) == type(a)

    print('Special function test passed...')


def test_operators():
    operators = ['+', '-', '*', '/', '**']
    # Base cases
    a = CartesianTensor(torch.ones(6, 3)).refine_names('Ka', ...)
    b = CartesianTensor(torch.ones(6, 3)).refine_names('Kb', ...)
    for op in operators:
        out = eval(f'a {op} b')
        assert out.shape == (6, 6, 3)
        assert out.names == ('Ka', 'Kb', None)
        assert type(out) == type(a)
    # Share one common name
    a = CartesianTensor(torch.ones(5, 6, 3)).refine_names('Ka','Kb', ...)
    b = CartesianTensor(torch.ones(6, 7, 3)).refine_names('Kb','Kc', ...)
    for op in operators:
        out = eval(f'a {op} b')
        assert out.shape == (5, 6, 7, 3)
        assert out.names == ('Ka', 'Kb', 'Kc', None)
        assert type(out) == type(a)

    a = CartesianTensor(torch.ones(5, 6, 3)).refine_names('Ka','Kb', ...)
    b = CartesianTensor(torch.ones(6, 7, 8, 3)).refine_names('Kb','Kc','Kd',...)
    for op in operators:
        out = eval(f'a {op} b')
        assert out.shape == (5, 6, 7, 8, 3)
        assert out.names == ('Ka', 'Kb', 'Kc', 'Kd', None)
        assert type(out) == type(a)

    # Share multiple common names
    a = CartesianTensor(torch.ones(5, 6, 8, 3)).refine_names('Ka','Kb','Kd', ...)
    b = CartesianTensor(torch.ones(6, 7, 8, 3)).refine_names('Kb','Kc','Kd',...)
    for op in operators:
        out = eval(f'a {op} b')
        assert out.shape == (5, 6, 7, 8, 3)
        assert out.names == ('Ka', 'Kb', 'Kc', 'Kd', None)
        assert type(out) == type(a)

    # Mixed CartesianTensor and normal tensor
    a = CartesianTensor(torch.ones(5, 6, 3)).refine_names('Ka','Kb', ...)
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

    a = CartesianTensor(torch.ones(5, 6)).refine_names('Ka','Kb')
    b = torch.randn(3)
    for op in operators:
        out = eval(f'a {op} b')
        assert out.shape == (5, 6, 3)
        assert out.names == ('Ka', 'Kb', None)
        out = eval(f'b {op} a')
        assert out.shape == (5, 6, 3)
        assert out.names == ('Ka', 'Kb', None)
        assert type(out) == type(a)

    a = CartesianTensor(torch.ones(5, 6)).refine_names('Ka','Kb')
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


def test_chain_operations():
    k_list = ['Ka', 'Kb', 'Kc', 'Kd', 'Ke', 'Kf']
    base_udf_shape = (4,)
    k_shape = (5,)
    base_tensor = torch.randn(4,) # Base udf tensor
    for k in k_list:
        base_tensor = base_tensor + CartesianTensor(torch.ones(k_shape)).refine_names(k,)
    assert base_tensor.shape == len(k_list) * k_shape + base_udf_shape
    assert base_tensor.names == tuple(k_list) + (None,)
    print("Chain operations test passed...")


test_operators()
test_special_functions()
test_chain_operations()