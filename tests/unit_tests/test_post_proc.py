import torch as t
import pytest
from alan.postproc import *


unweighted = t.randn((3, 4, 5), names=("plate_1", "N", None))

weights = t.rand((3,4,5), names=("plate_1", "K", None))
weights = weights / weights.sum("K", keepdim=True)
weighted = (t.randn((3,4,5), names=("plate_1", "K", None)), weights)

unweighted_dict = {'a': unweighted, 'b': unweighted}
weighted_dict = {'a': weighted, 'b': weighted}

def _test_map(f):
    res = f(unweighted)
    assert res.names == ("plate_1", "N", None)
    assert res.shape == (3, 4, 5)

    res = f(weighted)
    assert res[0].names == ("plate_1", "K", None)
    assert res[0].shape == (3, 4, 5)
    assert res[1].names == ("plate_1", "K", None)
    assert res[1].shape == (3, 4, 5)

    res = f(unweighted_dict)
    assert res['a'].names == ("plate_1", "N", None)
    assert res['a'].shape == (3,4, 5)
    assert res['b'].names == ("plate_1", "N", None)
    assert res['b'].shape == (3,4, 5)

def test_identity():
    _test_map(identity)
def test_square():
    _test_map(square)
def test_log():
    _test_map(log)

def _test_reduce(f):
    res = f(unweighted)
    assert res.names == ("plate_1", None)
    assert res.shape == (3, 5)

    res = f(weighted)
    assert res.names == ("plate_1", None)
    assert res.shape == (3, 5)

    res = f(unweighted_dict)
    assert res['a'].names == ("plate_1", None)
    assert res['a'].shape == (3,5)
    assert res['b'].names == ("plate_1", None)
    assert res['b'].shape == (3,5)

    res = f(weighted_dict)
    assert res['a'].names == ("plate_1", None)
    assert res['a'].shape == (3,5)
    assert res['b'].names == ("plate_1", None)
    assert res['b'].shape == (3,5)
    
def test_mean():
    _test_reduce(mean)

def test_var():
    _test_reduce(var)

def test_std():
    _test_reduce(std)

def test_ess():
    """
    It only makes sense to apply the ESS to a weighted sample.
    """
    res = ess(weighted)
    assert res.names == ("plate_1", None)
    assert res.shape == (3, 5)

    res = ess(weighted_dict)
    assert res['a'].names == ("plate_1", None)
    assert res['a'].shape == (3,5)
    assert res['b'].names == ("plate_1", None)
    assert res['b'].shape == (3,5)

    with pytest.raises(Exception, match="Trying to compute the ESS"):
        ess(unweighted_dict)


