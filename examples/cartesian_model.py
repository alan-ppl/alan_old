import torch as t
import torch.nn as nn
import alan
t.manual_seed(0)

def P(tr):
    scale = 0.1
    tr.sample('a', alan.Normal(t.zeros(3,), 1))
    tr.sample('b', alan.Normal(tr['a'] + 1, 1))
    tr.sample('c', alan.Normal(t.zeros(3,), 1))
    tr.sample('d', alan.Normal(tr['c'] + tr['b'], 1))
    tr.sample('obs', alan.Normal(tr['d'], 0.1))


class Q(alan.QModule):
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
        tr.sample('a', alan.Normal(self.m_a, self.log_s_a.exp()))
        tr.sample('b', alan.Normal(self.m_b, self.log_s_b.exp()))
        tr.sample('c', alan.Normal(self.m_c, self.log_s_c.exp()))
        tr.sample('d', alan.Normal(self.m_d, self.log_s_d.exp()))



data = alan.sample(P,varnames=('obs',))

model = alan.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-2)

K=10

print("K={}".format(K))
for i in range(1000):
    opt.zero_grad()
    elbo = model.elbo(K=K)
    (-elbo).backward()
    opt.step()

    if 0 == i%100:
        print(elbo.item())
