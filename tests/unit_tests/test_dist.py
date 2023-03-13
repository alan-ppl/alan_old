import torch as t
import alan
from functorch.dim import Dim

import pytest

d3 = Dim('d1', 3)
d4 = Dim('d1', 4)
d5 = Dim('d1', 5)

pt  = t.randn(())
p3  = t.randn(3)[d3]
p4  = t.randn(4)[d4]
p3_  = t.randn(3,6)[d3]
p34 = t.randn(3,4)[d3,d4]
p345 = t.randn(3,4,5)[d3,d4,d5]

def sample_lp_asserter(dist, result_ndim, result_dims, sample_dims=(), reparam=False):
    assert isinstance(result_dims, (tuple, list))

    sample = dist.sample(reparam, sample_dims=sample_dims)
    lp = dist.log_prob(sample)
   
    assert sample.ndim == result_ndim
    assert lp.ndim == 0
    if 0==len(result_dims):
        assert isinstance(sample, t.Tensor) 
        assert isinstance(lp, t.Tensor) 
    else:
        assert set(sample.dims) == set(result_dims)
        assert set(lp.dims)     == set(result_dims)

def test_python_params():
    dist = alan.Normal(1, 1)
    sample_lp_asserter(dist, 0, ())
    sample_lp_asserter(dist, 0, (d3,), sample_dims=(d3,))
    sample_lp_asserter(dist, 0, (d3,d4), sample_dims=(d3,d4))

    sample_lp_asserter(dist, 0, (),                           reparam=True)
    sample_lp_asserter(dist, 0, (d3,),   sample_dims=(d3,),   reparam=True)
    sample_lp_asserter(dist, 0, (d3,d4), sample_dims=(d3,d4), reparam=True)

def test_torchdim():
    dist = alan.Normal(p34, p3_.exp())
    sample_lp_asserter(dist, 1, (d3,d4))
    sample_lp_asserter(dist, 1, (d3,d4),    sample_dims=(d3,))
    sample_lp_asserter(dist, 1, (d3,d4,d5), sample_dims=(d3,d5))

    sample_lp_asserter(dist, 1, (d3,d4),                         reparam=True)
    sample_lp_asserter(dist, 1, (d3,d4),    sample_dims=(d3,d4), reparam=True)
    sample_lp_asserter(dist, 1, (d3,d4,d5), sample_dims=(d5,),   reparam=True)

def test_python_torchdim_params():
    dist = alan.Normal(p3_, 1)
    sample_lp_asserter(dist, 1, (d3,))
    sample_lp_asserter(dist, 1, (d3,), sample_dims=(d3,))
    sample_lp_asserter(dist, 1, (d3,d4), sample_dims=(d3,d4))

    sample_lp_asserter(dist, 1, (d3,),                        reparam=True)
    sample_lp_asserter(dist, 1, (d3,),   sample_dims=(d3,),   reparam=True)
    sample_lp_asserter(dist, 1, (d3,d4), sample_dims=(d3,d4), reparam=True)

def test_torchdim_torch_params():
    dist = alan.Normal(p3_, t.ones(6))
    sample_lp_asserter(dist, 1, (d3,))
    sample_lp_asserter(dist, 1, (d3,), sample_dims=(d3,))
    sample_lp_asserter(dist, 1, (d3,d4), sample_dims=(d3,d4))

    sample_lp_asserter(dist, 1, (d3,),                        reparam=True)
    sample_lp_asserter(dist, 1, (d3,),   sample_dims=(d3,),   reparam=True)
    sample_lp_asserter(dist, 1, (d3,d4), sample_dims=(d3,d4), reparam=True)

def test_python_torch_params():
    dist = alan.Normal(1, t.ones(6))
    sample_lp_asserter(dist, 1, ())
    sample_lp_asserter(dist, 1, (d3,), sample_dims=(d3,))
    sample_lp_asserter(dist, 1, (d3,d4), sample_dims=(d3,d4))

    sample_lp_asserter(dist, 1, (),                           reparam=True)
    sample_lp_asserter(dist, 1, (d3,),   sample_dims=(d3,),   reparam=True)
    sample_lp_asserter(dist, 1, (d3,d4), sample_dims=(d3,d4), reparam=True)

def test_kwargs():
    dist = alan.Categorical(probs=t.ones(3))
    sample_lp_asserter(dist, 0, ())
    sample_lp_asserter(dist, 0, (d3,), sample_dims=(d3,))
    sample_lp_asserter(dist, 0, (d3,d4), sample_dims=(d3,d4))

    dist = alan.Categorical(logits=t.zeros(3))
    sample_lp_asserter(dist, 0, ())
    sample_lp_asserter(dist, 0, (d3,), sample_dims=(d3,))
    sample_lp_asserter(dist, 0, (d3,d4), sample_dims=(d3,d4))

def test_parse():
    assert {'a': 1} == alan.dist.parse(('a',), (1,), {})
    assert {'a': 1} == alan.dist.parse(('a',), (), {'a': 1})

    assert {'a': 1, 'b': 2} == alan.dist.parse(('a', 'b'), (), {'a': 1, 'b': 2})
    assert {'a': 1, 'b': 2} == alan.dist.parse(('a', 'b'), (1,), {'b': 2})
    assert {'a': 1, 'b': 2} == alan.dist.parse(('a', 'b'), (1, 2), {})

    with pytest.raises(Exception, match='Unrecognised argument'):
        alan.dist.parse(('a', 'b'), (1, 2), {'c': 3})
    with pytest.raises(Exception, match='Too many arguments'):
        alan.dist.parse(('a', 'b'), (1, 2, 3), {})
    with pytest.raises(Exception, match='Multiple values'):
        alan.dist.parse(('a', 'b'), (1, 2), {'a': 1})

    with pytest.raises(Exception, match='Unrecognised argument'):
        alan.Normal(1, 2, un=3)
    with pytest.raises(Exception, match='Too many arguments'):
        alan.Normal(1, 2, 3)
    with pytest.raises(Exception, match='Multiple values'):
        alan.Normal(1, 2, scale=3)

def test_arg_sizes():
    mean = t.zeros(3)
    sqrt_cov = t.randn(3,3)
    cov = sqrt_cov @ sqrt_cov.mT

    dist = alan.MultivariateNormal(mean, cov)
    sample_lp_asserter(dist, 1, ())
    sample_lp_asserter(dist, 1, (d3,), sample_dims=(d3,))
    sample_lp_asserter(dist, 1, (d3,d4), sample_dims=(d3,d4))

    dist = alan.MultivariateNormal(mean[None, :].expand(4, -1)[d4], cov)
    sample_lp_asserter(dist, 1, (d4,))
    sample_lp_asserter(dist, 1, (d3,d4), sample_dims=(d3,))

    dist = alan.MultivariateNormal(mean[None].expand(4, -1)[d4], cov[None].expand(4, -1, -1)[d4])
    sample_lp_asserter(dist, 1, (d4,))
    sample_lp_asserter(dist, 1, (d3,d4), sample_dims=(d3,))

    with pytest.raises(Exception, match='should have dimension'):
        dist = alan.MultivariateNormal(t.ones(()), t.eye(3))

def test_reparam():
    dist = alan.Categorical(probs=t.ones(3))

    with pytest.raises(Exception, match='Trying to do reparam'):
        sample_lp_asserter(dist, 0, (), reparam=True)

