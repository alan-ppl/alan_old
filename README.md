[![Documentation Status](https://readthedocs.org/projects/alan-ppl/badge/?version=dev)](http://alan-ppl.readthedocs.io/en/latest/?badge=dev)

[Documentation](https://alan-ppl.readthedocs.io/en/latest/)

<!-- To get started:

```
pip install -e ./
```


Dependency:
- Torch 1.13
- Current version of Functorch: https://github.com/facebookresearch/functorch `pip install functorch`

Notes:
- approximate posterior should be independent of data
- On MacOs, you probably need to `export MACOSX_DEPLOYMENT_TARGET=10.9` before installing functorch

TODOs:
- document that you have to be _really_ careful with dimensions in your programme.
- document how to set the dimensions for data!
- More rigorous testing workflow and cases. (e.g. Using unit test framework like pytest)
- More examples for tpp. -->

Alan: Probabilistic Programming with Massively Parallel Importance Weighting
=====================================================

This library showcases Massively Parallel Importance Weighting in the context of Variational Inference based probabilistic programming. Using Importance Weighted Autoencoder (IWAE) and Reweighted Wake Sleep (RWS) as inference methods, for a graphical model with $n$ latent variables we can obtain $K^n$ proposals where $K$ is determined by the user. This improves inference performance and allows for...

## A Preview: Fitting a simple gaussian Model

```py
import torch as t
import torch.nn as nn
import tpp

def P(tr):
  '''
  Bayesian Gaussian Model
  '''
  a = t.zeros(5)
  tr.sample('mu', tpp.Normal(a, t.ones(5)))
  tr.sample('obs', tpp.MultivariateNormal(tr['mu'], t.eye(5)))

class Q(tpp.QModule):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(5,))
        self.log_s_mu = nn.Parameter(t.zeros(5,))

    def forward(self, tr):
        tr.sample('mu', tpp.Normal(self.m_mu, self.log_s_mu.exp()))

data = tpp.sample(P, varnames=('obs',))

model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=5
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(K)
    (-elbo).backward()
    opt.step()
```

Installation from Source
========================

```sh
git clone git@github.com:ThomasHeap/tpp.git
cd tpp
git checkout main  # master is pinned to the latest release
pip install -e . # pip install .[extras] for running some models in examples/
```

Defining Probabilistic Models
=============================

### Defining $P$

The probabilistic model is defined as a function `P(tr)` with argument `tr` that allows the tracer to collect samples of each latent variable. The usual structure of `P(tr)` is a series of `tr.sample('latent_name', tpp.dist(params))` calls telling the tracer to sample these latents, followed by `tr.sample('obs', tpp.dist(params))` for sampling the observations.

```py
def P(tr):
  tr.sample('first_latent', tpp.dist(params))
  tr.sample('second_latent', tpp.dist(params))
  ...
  tr.sample('obs', tpp.dist(params))
```
where `tpp.dist` is one of the distributions [listed below](#choice-of-distributions)

#### Plated and Grouped latents

![plated_model](./imgs/plated_model.png)

You can define the plated model above as so:
```py
sizes = {'plate_1':N, 'plate_2':M}
def P(tr):
    tr.sample('mu',   tpp.MultivariateNormal(t.zeros(5), t.eye(5)))
    tr.sample('phi', tpp.MultivariateNormal(tr['phi'], t.eye(5)), plate='plate_1')
    tr.sample('obs',   tpp.Normal(tr['phi'], 1), plate='plate_2')
```

and you can group latents together:
```py
def P(tr):
    tr.sample('mu',   tpp.MultivariateNormal(t.zeros(5), t.eye(5)), group='group_1')
    tr.sample('phi', tpp.MultivariateNormal(tr['phi'], t.eye(5)), group='group_1')
    tr.sample('psi', tpp.MultivariateNormal(t.zeros(5), t.eye(5)), group='group_2')
    tr.sample('obs',   tpp.Normal(tr['phi'] + tr['psi'], 1))
```

Combinations between the importance samples within groups of latents are not computed. For a model with $n$ latents, $m$ of which are grouped into $k$ groups $(k \leq m)$ there will be $K^{n - m + k}$ importance samples drawn.

<!-- #### Using weights and conditional information
We show in [Conditioning On Data](#using-pre-existing-data) how to load weights and conditional information. To use it in $P$ you do the following (assuming the key 'weights' is used): -->





### Defining $Q$

The proposal $Q$ is defined as a class inheriting `tpp.Q`. The learnable parameters are defined in `__init__` and in `forward` we define how the latents are sampled and interact.

```py
class Q(tpp.QModule):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(5))
        self.log_s_mu = nn.Parameter(t.zeros(5))

    def forward(self, tr):
        tr.sample('mu', tpp.Normal(self.m_mu, self.log_s_mu.exp()))
```
as with $P$ the call to sample can take a plate argument, or the learnable parameters can take the plate structure into account:

```py
sizes = {'plate_1':N, 'plate_2':M}
class Q(tpp.QModule):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(()))
        self.log_s_mu = nn.Parameter(t.zeros(()))

        self.m_phi = nn.Parameter(t.zeros((N)), names=('plate_1'))
        self.log_s_phi = nn.Parameter(t.zeros((N)), names=('plate_1'))

    def forward(self, tr):
        tr.sample('mu', tpp.Normal(self.m_mu, self.log_s_mu.exp()))
        tr.sample('phi', tpp.Normal(self.m_phi, self.log_s_phi.exp()))
```

### Choice of distributions
Choices for `tpp.dist` include most distributions listed in https://pytorch.org/docs/stable/distributions.html:

- `tpp.Bernoulli`
- `tpp.Beta`
- `tpp.Binomial`
- `tpp.Categorical`
- `tpp.Cauchy`
- `tpp.Chi2`
- `tpp.ContinuousBernoulli`
- `tpp.Exponential`
- `tpp.FisherSnedecor`
- `tpp.Gamma`
- `tpp.Geometric`
- `tpp.Gumbel`
- `tpp.HalfCauchy`
- `tpp.HalfNormal`
- `tpp.Kumaraswamy`
- `tpp.LKJCholesky`
- `tpp.Laplace`
- `tpp.LogNormal`
- `tpp.LowRankMultivariateNormal`
- `tpp.Multinomial`
- `tpp.MultivariateNormal`
- `tpp.NegativeBinomial`
- `tpp.Normal`
- `tpp.Pareto`
- `tpp.Poisson`
- `tpp.RelaxedBernoulli`
- `tpp.RelaxedOneHotCategorical`
- `tpp.StudentT`
- `tpp.Uniform`
- `tpp.VonMises`
- `tpp.Weibull`
- `tpp.Wishart`

Note: For models with discrete latents only the [RWS](#reweighted-wake-sleep) inference method is supported.

Covariates
===========

Having defined a probabilistic model and proposal you can provide data as so:

### Using the probabilistic model to provide data

If you don't already have data you can use $P$ to sample some:

```py
data = tpp.sample(P, varnames=('obs',))
model = tpp.Model(P, Q(), data)
```

### Using pre-existing data
If you already have some saved data you can load them (as PyTorch Tensors) as use them as follows:

```py
obs = loaded_obs.rename('plate_1', 'plate_2',...) #rename with whichever plate names you are using in P

#some weights
weights = loaded_weights.rename('plate_1', 'plate_2',...) #Or whichever plates are needed

data = {'obs':obs, 'weights':weights}

def P(tr):
    tr.sample('mu',   tpp.MultivariateNormal(t.zeros(5), t.eye(5)))
    tr.sample('obs',   tpp.Normal(tr['mu'] @ tr['weights'], 1))

### Define Q
...

model = tpp.Model(P, Q(), data)

### Train Q
...
```

A good example of this is in [the radon model](./examples/radon/randon_model_unif.py)

Inference Methods
=================

We support two objective functions for inference. All methods require the user to specify the number $K$ of samples to be drawn for each latent (or group of latents).

## Evidence Lower Bound (IWAE)
In the training loop:
```py
K=5
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(K)
    (-elbo).backward()
    opt.step()
```

## Reweighted Wake Sleep
We support wake theta and phi loss. (See [The paper](link_to_the_paper) for more information).
```py
K=5
for i in range(10000):
    opt.zero_grad()
    wake_theta_loss, wake_phi_loss = model.rws(K=K)
    (-wake_theta_loss + wake_phi_loss).backward()
    opt.step()
```

Calculating Moments
===================

Given a proposal (trained or untrained) we can draw importance weighting moments from the posterior:

```py
from tpp.postproc import mean

### Define P and Q and data

model = tpp.Model(P, Q(), data)

### Optionally train model

#Compute the mean with 10000 importance weights
mean(model.weights(10000))
```

Moments supported:

- `mean` - Mean
- `mean2` - Squared mean
- `p_lower(value)` - Proportion of samples lower than `value`
- `p_higher(value)` - Proportion of samples higher than `value`
- `var` - Variance
- `std` - Standard Deviation
- `ess` - Effective Sample Size
- `stderr_mean` - Standard error of the mean
- `stderr_mean2` - Standard error of the squared mean
- `stderr_p_lower` - Standard error of p_lower
- `stderr_p_higher` - Standard error of p_higher

Calculating predictive log likelihood
=====================================

We can calculate predictive log likelihood for a model as follows:

```py
### Define P and Q and data

model = tpp.Model(P, Q(), data)

### train model

#Compute the mean with 10000 importance weights
model.predictive_ll(K, num_importance_weights, data_all={"obs": obs})
```

<!--
have to note that data_all must be larger than the training data, you have to concatenate training and testing data_all
Also note that pred_lls can be calculated just with training data

 -->


Timeseries
==========

Tips for dealing with out of memory errors
==========================================

It is better for memory usage if your model looks like:

![good_practice](./imgs/Good_practice.png)

rather than:

![bad_practice](./imgs/bad_practice.png)

i.e Try not to have latent variables 'skip' plates.
