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
    assert type(out) == type(a)

    # Test outer product
    a = CartesianTensor(torch.ones(5, 6, 3)).refine_names('Ka','Kb', ...)
    b = CartesianTensor(torch.ones(6, 7, 4)).refine_names('Kb','Kc', ...)
    out = torch.outer(a, b)
    assert out.shape == (5, 6, 7, 3, 4)
    assert type(out) == type(a)

    print('Special function test passed...')


def test_operators():
    operators = ['+', '-', '*', '/', '**']
    a = CartesianTensor(torch.ones(6, 3)).refine_names('Ka', ...)
    b = CartesianTensor(torch.ones(6, 3)).refine_names('Kb', ...)
    for op in operators:
        out = eval(f'a {op} b')
        assert out.shape == (6, 6, 3)
        assert type(out) == type(a)

    a = CartesianTensor(torch.ones(5, 6, 3)).refine_names('Ka','Kb', ...)
    b = CartesianTensor(torch.ones(6, 7, 3)).refine_names('Kb','Kc', ...)
    for op in operators:
        out = eval(f'a {op} b')
        assert out.shape == (5, 6, 7, 3)
        assert type(out) == type(a)

    a = CartesianTensor(torch.ones(5, 6, 3)).refine_names('Ka','Kb', ...)
    b = CartesianTensor(torch.ones(6, 7, 8, 3)).refine_names('Kb','Kc','Kd',...)
    for op in operators:
        out = eval(f'a {op} b')
        assert out.shape == (5, 6, 7, 8, 3)
        assert type(out) == type(a)

    a = CartesianTensor(torch.ones(5, 6, 3)).refine_names('Ka','Kb', ...)
    b = torch.randn(3)
    for op in operators:
        out = eval(f'a {op} b')
        assert out.shape == (5, 6, 3)
        assert type(out) == type(a)
        out = eval(f'b {op} a')
        assert out.shape == (5, 6, 3)
        assert type(out) == type(a)

    a = CartesianTensor(torch.ones(5, 6)).refine_names('Ka','Kb')
    b = torch.randn(3)
    for op in operators:
        out = eval(f'a {op} b')
        assert out.shape == (5, 6, 3)
        out = eval(f'b {op} a')
        assert out.shape == (5, 6, 3)
        assert type(out) == type(a)

    a = CartesianTensor(torch.ones(5, 6)).refine_names('Ka','Kb')
    b = torch.randn(3, 2, 3)
    for op in operators:
        out = eval(f'a {op} b')
        assert out.shape == (5, 6, 3, 2, 3)
        out = eval(f'b {op} a')
        assert out.shape == (5, 6, 3, 2, 3)
        assert type(out) == type(a)
    print("Operator test passed...")


test_operators()
test_special_functions()