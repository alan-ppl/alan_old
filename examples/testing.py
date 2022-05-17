import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi

def P(tr):
  '''
  Bayesian Heirarchical Model
  Gaussian with Wishart Prior on precision
  '''
  a = t.zeros(5,)
  tr['mu'] = tpp.Normal(a, 1)
  tr['obs'] = tpp.Normal(tr['mu'], 1)



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(5,))
        #self.m_mu = nn.Parameter(t.tensor([[-0.2655, -0.5713, -0.1689,  0.4844, -1.0193]]))


        self.log_s_mu = nn.Parameter(t.zeros(5,))
        #self.log_s_mu = nn.Parameter(t.tensor([[0.1667, 0.1667, 0.1667, 0.1667, -0.1667]]))



    def forward(self, tr):
        tr['mu'] = tpp.Normal(self.m_mu, self.log_s_mu.exp())




data = tpp.sample(P, "obs")
print('data')
print(data)
model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

trq = TraceSampleLogQ(K=10, data=data)
model.Q(trq)



trp = TraceLogP(trq.sample, model.data)
model.P(trp)

print('trp')
print(trp.log_prob())
