Fitting a simple Gaussian Model using Reweighted Wake-Sleep
===========================================================
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

