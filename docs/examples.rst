Examples
========

Installation from Source
************************

.. code-block:: bash

  git clone git@github.com:ThomasHeap/tpp.git
  cd tpp
  git checkout main
  pip install . # pip install .[extras] for running some models in examples/



Fitting a simple gaussian Model
*******************************
.. code-block:: python

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
