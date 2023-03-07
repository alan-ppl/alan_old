Examples
========

Installation from Source
************************

.. code-block:: bash

  git clone git@github.com:alan-ppl/alan.git
  cd alan
  git checkout main
  pip install . # pip install .[extras] for running some models in examples/



Fitting a simple Gaussian Model using Variational Inference
***********************************************************
.. code-block:: python

  import torch as t
  import torch.nn as nn
  import alan

  def P(tr):
    '''
    Bayesian Gaussian Model
    '''
    a = t.zeros(5)
    tr('mu', alan.Normal(a, t.ones(5)))
    tr('obs', alan.MultivariateNormal(tr['mu'], t.eye(5)))


  class Q(alan.AlanModule):
      def __init__(self):
          super().__init__()
          self.m_mu = nn.Parameter(t.zeros(5,))
          self.log_s_mu = nn.Parameter(t.zeros(5,))

      def forward(self, tr):
          tr('mu', alan.Normal(self.m_mu, self.log_s_mu.exp())) #, plate="plate_1")

  data = alan.sample_prior(varnames='obs')

  cond_model = alan.Model(P, Q()).condition(data=data)

  opt = t.optim.Adam(model.parameters(), lr=1E-3)

  K=10
  print("K={}".format(K))
  for i in range(20000):
      opt.zero_grad()
      elbo = cond_model.sample_perm(K, True).elbo()
      (-elbo).backward()
      opt.step()

      if 0 == i%1000:
          print(elbo.item())

Fitting a simple Gaussian Model using Reweighted Wake-Sleep
***********************************************************
.. code-block:: python

  import torch as t
  import torch.nn as nn
  import alan

  def P(tr):
    '''
    Bayesian Gaussian Model
    '''
    a = t.zeros(5)
    tr('mu', alan.Normal(a, t.ones(5)))
    tr('obs', alan.MultivariateNormal(tr['mu'], t.eye(5)))


  class Q(alan.AlanModule):
      def __init__(self):
          super().__init__()
          self.m_mu = nn.Parameter(t.zeros(5,))
          self.log_s_mu = nn.Parameter(t.zeros(5,))

      def forward(self, tr):
          tr('mu', alan.Normal(self.m_mu, self.log_s_mu.exp())) #, plate="plate_1")

  data = alan.sample_prior(varnames='obs')

  cond_model = alan.Model(P, Q()).condition(data=data)

  opt = t.optim.Adam(model.parameters(), lr=1E-3)

  K=10
  print("K={}".format(K))
  for i in range(20000):
      opt.zero_grad()
      p_obj, q_obj = cond_model.sample_perm(K, True).rws()
      (-q_obj).backward()
      opt.step()

      if 0 == i%1000:
          print(q_obj.item())


Plated model
************
.. code-block:: python

  import torch as t
  import torch.nn as nn
  import alan
  t.manual_seed(0)

  J = 2
  M = 3
  N = 4
  platesizes = {'plate_1': J, 'plate_2': M, 'plate_3': N}
  def P(tr):
      tr('a',   alan.Normal(tr.zeros(()), 1))
      tr('b',   alan.Normal(tr['a'], 1))
      tr('c',   alan.Normal(tr['b'], 1), plates='plate_1')
      tr('d',   alan.Normal(tr['c'], 1), plates='plate_2')
      tr('obs', alan.Normal(tr['d'], 1), plates='plate_3')

  class Q(alan.AlanModule):
      def __init__(self):
          super().__init__()
          self.m_a = nn.Parameter(t.zeros(()))
          self.w_b = nn.Parameter(t.zeros(()))
          self.b_b = nn.Parameter(t.zeros(()))

          self.w_c = nn.Parameter(t.zeros((J,), names=('plate_1',)))
          self.b_c = nn.Parameter(t.zeros((J,), names=('plate_1',)))

          self.w_d = nn.Parameter(t.zeros((M, J), names=('plate_2','plate_1')))
          self.b_d = nn.Parameter(t.zeros((M, J), names=('plate_2','plate_1')))

          self.log_s_a = nn.Parameter(t.zeros(()))
          self.log_s_b = nn.Parameter(t.zeros(()))
          self.log_s_c = nn.Parameter(t.zeros((J,), names=('plate_1',)))
          self.log_s_d = nn.Parameter(t.zeros((M,J), names=('plate_2','plate_1')))


      def forward(self, tr):
          tr('a', alan.Normal(self.m_a, self.log_s_a.exp()))

          mean_b = self.w_b * tr['a'] + self.b_b
          tr('b', alan.Normal(mean_b, self.log_s_b.exp()))

          mean_c = self.w_c * tr['b'] + self.b_c
          tr('c', alan.Normal(mean_c, self.log_s_c.exp()))

          mean_d = self.w_d * tr['c'] + self.b_d
          tr('d', alan.Normal(mean_d, self.log_s_d.exp()))

  model = alan.Model(P, Q())
  data = model.sample_prior(varnames='obs', platesizes={'plate_3': N})
  cond_model = alan.Model(P, Q()).condition(data=data)

  opt = t.optim.Adam(cond_model.parameters(), lr=1E-3)

  K=10
  print("K={}".format(K))
  for i in range(20000):
      opt.zero_grad()
      elbo = cond_model.sample_cat(K, True).elbo()
      (-elbo).backward()
      opt.step()

      if 0 == i%1000:
          print(elbo.item())

Fast inference with exponential family
**************************************
.. code-block:: python

  import torch as t
  import torch.nn as nn
  import alan
  t.manual_seed(0)

  J = 2
  M = 3
  N = 4
  platesizes = {'plate_1': J, 'plate_2': M, 'plate_3': N}
  def P(tr):
      tr('a',   alan.Normal(t.zeros(()), 1))
      tr('b',   alan.Normal(tr['a'], 1))
      tr('c',   alan.Normal(tr['b'], 1), plates='plate_1')
      tr('d',   alan.Normal(tr['c'], 1), plates='plate_2')
      tr('obs', alan.Normal(tr['d'], 0.01), plates='plate_3')

  class Q(alan.AlanModule):
      def __init__(self):
          super().__init__()
          self.Na = alan.MLNormal()
          self.Nb = alan.MLNormal()
          self.Nc = alan.MLNormal({'plate_1': J})
          self.Nd = alan.MLNormal({'plate_1': J, 'plate_2': M})

      def forward(self, tr):
          tr('a',   self.Na())
          tr('b',   self.Nb())
          tr('c',   self.Nc())
          tr('d',   self.Nd())

  data = alan.Model(P).sample_prior(platesizes=platesizes, varnames='obs')

  K = 100
  T = 40
  lr = 0.1

  t.manual_seed(0)
  q = Q()
  m1 = alan.Model(P, q).condition(data=data)
  for i in range(T):
      sample = m1.sample_same(K, reparam=False)
      print(sample.elbo().item())
      m1.update(lr, sample)
