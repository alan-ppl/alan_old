import torch as t
import torch.nn as nn
import tpp
t.manual_seed(0)

def P(tr):
    scale = 0.1
    tr.sample('a', tpp.Normal(t.zeros(3,), 1))
    tr.sample('b', tpp.Normal(tr['a'] + 1, 1))
    tr.sample('c', tpp.Normal(t.zeros(3,), 1))
    tr.sample('d', tpp.Normal(tr['c'] + tr['b'], 1))
    tr.sample('obs', tpp.Normal(tr['d'], 0.1))


class Q(tpp.Q):
    def __init__(self):
        super().__init__()
        self.reg_param('m_a', t.zeros(3,))
        self.reg_param('m_b', t.zeros(3,))
        self.reg_param('m_c', t.zeros(3,))
        self.reg_param('m_d', t.zeros(3,))

        self.reg_param('log_s_a', t.zeros(3,))
        self.reg_param('log_s_b', t.zeros(3,))
        self.reg_param('log_s_c', t.zeros(3,))
        self.reg_param('log_s_d', t.zeros(3,))


    def forward(self, tr):
        tr.sample('a', tpp.Normal(self.m_a, self.log_s_a.exp()))
        tr.sample('b', tpp.Normal(self.m_b, self.log_s_b.exp()))
        tr.sample('c', tpp.Normal(self.m_c, self.log_s_c.exp()))
        tr.sample('d', tpp.Normal(self.m_d, self.log_s_d.exp()))



data = tpp.sample(P,varnames=('obs',))

model = tpp.Model(P, Q(), data)

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
