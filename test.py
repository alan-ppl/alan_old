import torch as t
import torch.nn as nn
import tpp

def P(tr): 
    tr.set_names('a', 'b')
    tr['a'] = tpp.Normal(tr.zeros(()), 1)
    tr['b'] = tpp.Normal(tr['a'], 1)
    tr.add_remove_names(('c',), ('a',))
    tr['c'] = tpp.Normal(tr['b'], 1, sample_shape=3, sample_names='plate_a')
    tr['obs'] = tpp.Normal(tr['c'], 1, sample_shape=5, sample_names='plate_b')


class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_a = nn.Parameter(t.zeros(()))
        self.m_b = nn.Parameter(t.zeros(()))
        self.m_c = nn.Parameter(t.zeros((3,), names=('plate_a',)))

        self.log_s_a = nn.Parameter(t.zeros(()))
        self.log_s_b = nn.Parameter(t.zeros(()))
        self.log_s_c = nn.Parameter(t.zeros((3,), names=('plate_a',)))

    def forward(self, tr):
        tr['a'] = tpp.Normal(tr.pad(self.m_a), tr.pad(self.log_s_a.exp()))
        tr['b'] = tpp.Normal(tr.pad(self.m_b), tr.pad(self.log_s_b.exp()))
        tr['c'] = tpp.Normal(tr.pad(self.m_c), tr.pad(self.log_s_c.exp()))

data = tpp.sample(P, "obs")

model = tpp.Model(P, Q(), data)
elbo = model.elbo(K=1)

opt = t.optim.Adam(model.parameters(), lr=1E-2)
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(K=1)
    (-elbo).backward()
    opt.step()

    if 0 == i%100:
        print(elbo.item())
    
