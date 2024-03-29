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

def _test_prior(P=P, platesizes=platesizes, inputs={}):
    model = alan.Model(P, P)
    data = model.sample_prior(varnames='obs', platesizes=platesizes, inputs=inputs)



#### We use a variable tr['a'], before its sampled.
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


        
#### We sample a single variable twice
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



#### A latent variable appears in P/Q but not in Q/P
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



#### Data not provided.
def test_missing_data():
    with pytest.raises(Exception, match="provided, but not specified in P"):
        _test_ap(P=Q)



#### Required platesizes are not provided
def test_no_plate_size_sample():
    with pytest.raises(Exception, match="size of plate"):
        _test_prior(platesizes={'plate_2':4, 'plate_3':5})
    with pytest.raises(Exception, match="size of plate"):
        _test_prior(platesizes={})



#### A single plate is provided with inconsistent sizes from inputs and platesizes
#### (which is a default argument in _test_prior
def prog_input_inconsistent_plate(tr, inp):
    tr('a',   alan.Normal(tr.zeros(()), 1))
    tr('b',   alan.Normal(tr['a'], 1))
    tr('c',   alan.Normal(tr['b'], 1), plates='plate_1')
    tr('d',   alan.Normal(tr['c'], 1), plates='plate_2')
    tr('obs', alan.Normal(tr['d'], 1), plates='plate_3')
inp = t.ones(6, names=('plate_1',))
def test_input_inconsistent_plate():
    with pytest.raises(Exception, match="Mismatch in sizes for"):
        _test_prior(P=prog_input_inconsistent_plate, inputs={'inp': inp})


#### Try to analytically sum over a continuous distribution
def prog_sum_discrete_normal(tr):
    tr('a',   alan.Normal(tr.zeros(()), 1), sum_discrete=True)
    tr('b',   alan.Normal(tr['a'], 1))
    tr('c',   alan.Normal(tr['b'], 1), plates='plate_1')
    tr('d',   alan.Normal(tr['c'], 1), plates='plate_2')
    tr('obs', alan.Normal(tr['d'], 1), plates='plate_3')
def test_sum_discrete_normal_P():
    with pytest.raises(Exception, match="Can only sum over"):
        _test_ap(P=prog_sum_discrete_normal)



#### Try to analytically sum over a random variable, when we've
#### already sampled that variable in Q
def P_sum_discrete_inQ(tr):
    tr('a', alan.Bernoulli(0.5))
def Q_sum_discrete_inQ(tr):
    tr('a', alan.Bernoulli(0.5), sum_discrete=True)
def test_sum_discrete_inQ():
    with pytest.raises(Exception, match="We don't need an approximate posterior if"):
        _test_ap(P=P_sum_discrete_inQ, Q=Q_sum_discrete_inQ)

