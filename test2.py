from tpp.prob_prog import Trace, TraceLogP
import torch as t
import torch.nn as nn
import tpp

def P(tr): 
    scale = 0.1
    tr['a'] = tpp.Normal(t.zeros(3,), 1)
    tr['b'] = tpp.Normal(tr['a'] + 1, 1)
    tr['c'] = tpp.Normal(t.zeros(3,), 1)
    tr['d'] = tpp.Normal(tr['c'] + tr['b'], 1)
    tr['obs'] = tpp.Normal(tr['d'], 0.1)


class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_a = nn.Parameter(t.zeros(3,))
        self.m_b = nn.Parameter(t.zeros(3,))
        self.m_c = nn.Parameter(t.zeros(3,))
        self.m_d = nn.Parameter(t.zeros(3,))

        self.log_s_a = nn.Parameter(t.zeros(3,))
        self.log_s_b = nn.Parameter(t.zeros(3,))
        self.log_s_c = nn.Parameter(t.zeros(3,))
        self.log_s_d = nn.Parameter(t.zeros(3,))

    def forward(self, tr):
        tr['a'] = tpp.Normal(self.m_a, self.log_s_a.exp())
        tr['b'] = tpp.Normal(self.m_b, self.log_s_b.exp())
        tr['c'] = tpp.Normal(self.m_c, self.log_s_c.exp())
        tr['d'] = tpp.Normal(self.m_d, self.log_s_d.exp())

data = tpp.sample(P, "obs")

model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-2)
    
print("K=10")
for i in range(1000):
    opt.zero_grad()
    elbo = model.elbo(K=10)
    (-elbo).backward()
    opt.step()

    if 0 == i%100:
        print(elbo.item())