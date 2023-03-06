Examples
========

Installation from Source
************************

.. code-block:: bash

  git clone git@github.com:alan-ppl/alan.git
  cd alan
  git checkout main
  pip install . # pip install .[extras] for running some models in examples/



Fitting a simple gaussian Model
*******************************
.. code-block:: python

  import torch as t
  import torch.nn as nn
  import alan

  def P(tr):
    '''
    Bayesian Gaussian Model
    '''
    a = t.zeros(5)
    tr.sample('mu', alan.Normal(a, t.ones(5))) #, plate="plate_1")
    tr.sample('obs', alan.MultivariateNormal(tr['mu'], t.eye(5)))


  class Q(alan.QModule):
      def __init__(self):
          super().__init__()
          self.m_mu = nn.Parameter(t.zeros(5,))
          self.log_s_mu = nn.Parameter(t.zeros(5,))

      def forward(self, tr):
          tr.sample('mu', alan.Normal(self.m_mu, self.log_s_mu.exp())) #, plate="plate_1")

  data = alan.sample(P, varnames=('obs',))

  model = tpp.Model(P, Q(), data)

  opt = t.optim.Adam(model.parameters(), lr=1E-3)

  K=5
  for i in range(10000):
      opt.zero_grad()
      elbo = model.elbo(K)
      (-elbo).backward()
      opt.step()
