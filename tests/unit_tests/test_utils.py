import pytest
pp = pytest.param
import alan
import torch as t
from functorch.dim import Dim

from alan.utils import *

d4 = Dim('d4', 4)
d5 = Dim('d5', 5)
d6 = Dim('d6', 6)

d4_dict = {'d4': d4}
d45_dict = {**d4_dict, 'd5': d5}
d456_dict = {**d45_dict, 'd6': d6}

def assert_shape_dims(x, shape, dims):
    assert set(generic_dims(x)) == set(dims)
    assert x.shape == shape

def test_sum_non_dim():
    f = sum_non_dim
    assert_shape_dims(f(t.ones(4,5,1)[d4,d5]), (), (d4, d5))
    assert_shape_dims(f(t.ones(4,5)[d4,d5]),   (), (d4, d5))
    assert_shape_dims(f(t.ones(4,5)),          (), ())
    
    assert f(1) == 1
    
def test_sum_dims():
    f = sum_dims
    assert_shape_dims(f(t.ones(4,5)[d4,d5], ()),      (), (d4, d5))
    assert_shape_dims(f(t.ones(4,5)[d4,d5], (d4,)),   (), (d5,))
    assert_shape_dims(f(t.ones(4,5)[d4,d5], (d4,d5)), (), ())

    assert_shape_dims(f(t.ones(4,5,1)[d4,d5], ()),      (1,), (d4, d5))
    assert_shape_dims(f(t.ones(4,5,1)[d4,d5], (d4,)),   (1,), (d5,))
    assert_shape_dims(f(t.ones(4,5,1)[d4,d5], (d4,d5)), (1,), ())

    assert_shape_dims(max_dims(t.ones(4)[d4], (d4,)), (), ())
    assert_shape_dims(min_dims(t.ones(4)[d4], (d4,)), (), ())

    assert 1 == f(1, ())

    with pytest.raises(Exception, match='dims must be a list or tuple'):
        f(t.ones(4,5,1)[d4,d5], d4)
    with pytest.raises(Exception, match="dims provided that aren't in x"):
        1 == f(1, (d4,))

def test_is_dimtensor():
    assert is_dimtensor(t.ones(4,5)[d4])
    assert not is_dimtensor(t.ones(4,5))
    assert not is_dimtensor(1)

def test_unify_dims():
    assert [] == unify_dims([t.ones(4,5)])
    assert [d4, d5, d6] == unify_dims([t.ones(4,5)[d4,d5], t.ones(5)[d5], t.ones(4,6)[d4,d6], t.ones(3)])

def test_generic_ndim():
    assert 2 == generic_ndim(t.ones(3,4))
    assert 1 == generic_ndim(t.ones(4,3)[d4])
    assert 0 == generic_ndim(534)

def test_generic_dims():
    assert () == generic_dims(t.ones(3,4))
    assert (d4,) == generic_dims(t.ones(4,5)[d4])
    assert (d4,d5) == generic_dims(t.ones(4,5)[d4,d5])
    assert () == generic_dims(534)

def test_generic_order():
    assert_shape_dims(generic_order(t.ones(4,5),        ()),      (4,5), ())
    assert_shape_dims(generic_order(t.ones(4,5)[d4],    (d4,)),   (4,5), ())
    assert_shape_dims(generic_order(t.ones(4,5)[d4,d5], ()),      (),    (d4,d5))
    assert_shape_dims(generic_order(t.ones(4,5)[d4,d5], (d5,)),   (5,),  (d4,))
    assert_shape_dims(generic_order(t.ones(4,5)[d4,d5], (d4,d5)), (4,5), ())

    with pytest.raises(Exception, match='dims must be a list or tuple'):
        generic_order(t.ones(4,5), d4)

#TODO: generic_getitem
#TODO: generic_setitem

def test_ordered_unique():
    assert [1,2,3,4] == ordered_unique((1,1,2,1,3,1,2,4))
    with pytest.raises(Exception, match='ls must be a list or tuple'):
        ordered_unique(1)

def test_partition_tensors():
    t1 = t.ones(4,5)[d4,d5]
    t2 = t.ones(5)[d5]
    t3 = t.ones(5,7)[d5]
    t4 = t.ones(4)[d4]
    t5 = t.ones(4,7)[d4]
    t6 = t.ones(3)
    t7 = 1
    has_dim, no_dim = partition_tensors([t1,t2,t3,t4,t5,t6,t7], d4)
    assert set(has_dim) == {t1, t4, t5}
    assert set(no_dim) == {t2, t3, t6, t7}

def test_singleton_order():
    assert_shape_dims(singleton_order(t.ones(4, 5)[d4], [d4,d5,d6]), (4,1,1,5), ())
    assert_shape_dims(singleton_order(t.ones(4, 5)[d4,d5], [d6,d5]), (1,5), (d4,))

def test_dim2named_tensor():
    assert ('d4', 'd5', None) == dim2named_tensor(t.ones(4,5,6)[d4,d5]).names
    assert ('d4', 'd5') == dim2named_tensor(t.ones(4,5)[d4,d5]).names

def test_named2dim_tensor():
    assert_shape_dims(named2dim_tensor(d456_dict, t.ones((4,5), names=('d4','d5'))), (), (d4, d5))

    with pytest.raises(Exception, match='No torchdim dimension'):
        named2dim_tensor(d456_dict, t.ones((4,5), names=('a','d5')))

def test_logmmexp():
    t.manual_seed(0)
    log_X = t.randn(4,5)
    log_Y = t.randn(5,6)
    X = log_X.exp()
    Y = log_Y.exp()
    XY = X@Y
    log_XY = XY.log()

    assert t.isclose(log_XY, logmmexp(log_X[d4, d5], log_Y[d5, d6], d5).order(d4, d6)).all()

def test_chain_logmmexp():
    t.manual_seed(0)
    K = 2
    log_X1 = t.randn(K,K)
    log_X2 = t.randn(K,K)
    log_X3 = t.randn(K,K)
    log_X4 = t.randn(K,K)
    X1, X2, X3, X4 = log_X1.exp(), log_X2.exp(), log_X3.exp(), log_X4.exp()
    X1234 = X1@X2@X3@X4
    log_X1234 = X1234.log()

    Tdim = Dim('Tdim', 4)
    Kprev = Dim('Tprev', 2)
    Kcurr = Dim('Tprev', 2)
    ms = t.stack([log_X1, log_X2, log_X3, log_X4], 0)[Tdim, Kprev, Kcurr]

    assert t.isclose(log_X1234, chain_logmmexp(ms, Tdim, Kprev, Kcurr).order(Kprev, Kcurr)).all()


def test_torchdim_einsum_reduce():
    lX = t.randn(4,5,6)[d4,d5,d6]
    lY = t.randn(4,5)[d4,d5]
    lZ = t.randn(6)[d6]
    X,Y,Z = lX.exp(), lY.exp(), lZ.exp()

    XYZ = sum_dims(X*Y*Z, (d4,d5,d6))
    XYZd = XYZ / d4.size / d5.size / d6.size
    assert t.isclose(XYZ, torchdim_einsum((X,Y,Z), (d4,d5,d6))).all()
    assert t.isclose(XYZd.log(), reduce_Ks((lX,lY,lZ), (d4,d5,d6))).all()

    XYZ = sum_dims(X*Y*Z, (d4,d5))
    XYZd = XYZ / d4.size / d5.size
    assert t.isclose(XYZ, torchdim_einsum((X,Y,Z), (d4,d5))).order(d6).all()
    assert t.isclose(XYZd.log(), reduce_Ks((lX,lY,lZ), (d4,d5))).order(d6).all()
