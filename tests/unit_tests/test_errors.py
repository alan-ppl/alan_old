import alan
import pytest
import torch as t


def P(tr):
    tr('a',   alan.Normal(tr.zeros(()), 1))
    tr('b',   alan.Normal(tr['a'], 1))
    tr('c',   alan.Normal(tr['b'], 1), plates='plate_1')
    tr('d',   alan.Normal(tr['c'], 1), plates='plate_2')
    tr('obs', alan.Normal(tr['d'], 1), plates='plate_3')

def Q(tr):
    tr('a',   alan.Normal(tr.zeros(()), 1))
    tr('b',   alan.Normal(tr['a'], 1))
    tr('c',   alan.Normal(tr['b'], 1), plates='plate_1')
    tr('d',   alan.Normal(tr['c'], 1), plates='plate_2')

model = alan.Model(P, Q)
platesizes={'plate_1':3, 'plate_2':4, 'plate_3':5}
data = model.sample_prior(varnames='obs', platesizes=platesizes)

def _test_ap(P=P, Q=Q, data=data, inputs={}):
    model = alan.Model(P, Q)
    cond_model = model.condition(data=data, inputs=inputs)
    cond_model.sample_cat(10)

def _test_prior(P=P, inputs={}):
    model = alan.Model(P, P)
    data = model.sample_prior(varnames='obs', platesizes=platesizes, inputs=inputs)



def prog_use_before_define(tr):
    tr['a']
def test_use_before_define_P():
    with pytest.raises(Exception, match="but a not present in data"):
        _test_ap(P=prog_use_before_define)
def test_use_before_define_Q():
    with pytest.raises(Exception, match="but a not present in data"):
        _test_ap(Q=prog_use_before_define)
def test_use_before_define_prior():
    with pytest.raises(Exception, match="but a not present in data"):
        _test_prior(P=prog_use_before_define)


        
def prog_sample_twice(tr):
    tr('a',   alan.Normal(tr.zeros(()), 1))
    tr('a',   alan.Normal(tr.zeros(()), 1))
def test_sample_twice_P():
    with pytest.raises(Exception, match="Trying to sample"):
        _test_ap(P=prog_sample_twice)
def test_sample_twice_Q():
    with pytest.raises(Exception, match="Trying to sample"):
        _test_ap(Q=prog_sample_twice)
def test_sample_twice_prior():
    with pytest.raises(Exception, match="Trying to sample"):
        _test_prior(P=prog_sample_twice)



def prog_missing_latent(tr):
    tr('a',   alan.Normal(tr.zeros(()), 1))
    tr('b',   alan.Normal(tr['a'], 1))
    tr('d',   alan.Normal(tr['b'], 1), plates=('plate_1', 'plate_2'))
    tr('obs', alan.Normal(tr['d'], 1), plates='plate_3')
def test_missing_latent_P():
    with pytest.raises(Exception, match="sampled in Q but not present in P"):
        _test_ap(P=prog_missing_latent)
def test_missing_latent_Q():
    with pytest.raises(Exception, match="Trying to compute log-prob for"):
        _test_ap(Q=prog_missing_latent)



def test_missing_data():
    with pytest.raises(Exception, match="provided, but not specified in P"):
        _test_ap(P=Q)




