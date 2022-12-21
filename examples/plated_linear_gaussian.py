import torch as t
import torch.nn as nn
import tpp
t.manual_seed(0)

J = 2
M = 3
N = 4
sizes = {'plate_1': J, 'plate_2': M, 'plate_3': N}
def P(tr):
    tr.sample('a',   tpp.Normal(t.zeros(()), 1))
    tr.sample('b',   tpp.Normal(tr['a'], 1))
    tr.sample('c',   tpp.Normal(tr['b'], 1), plate='plate_1')
    tr.sample('d',   tpp.Normal(tr['c'], 1), plate='plate_2')
    tr.sample('obs', tpp.Normal(tr['d'], 1), plate='plate_3')

#def Q(tr):
#    tr.sample('a',   tpp.Normal(t.zeros(()), 1))
#    tr.sample('b',   tpp.Normal(tr['a'], 1))
#    tr.sample('c',   tpp.Normal(tr['b'], 1), plate='plate_1')
#    tr.sample('d',   tpp.Normal(tr['c'], 1), plate='plate_2')

class Q(tpp.Q):
    def __init__(self):
        super().__init__()
        self.reg_param("m_a", t.zeros(()))
        self.reg_param("w_b", t.zeros(()))
        self.reg_param("b_b", t.zeros(()))

        self.reg_param("w_c", t.zeros((J,)), ['plate_1'])
        self.reg_param("b_c", t.zeros((J,)), ['plate_1'])

        self.reg_param("w_d", t.zeros((M, J)), ['plate_2','plate_1'])
        self.reg_param("b_d", t.zeros((M, J)), ['plate_2','plate_1'])

        self.reg_param("log_s_a", t.zeros(()))
        self.reg_param("log_s_b", t.zeros(()))
        self.reg_param("log_s_c", t.zeros((J,)), ['plate_1'])
        self.reg_param("log_s_d", t.zeros((M,J)), ['plate_2','plate_1'])


    def forward(self, tr):
        tr.sample('a', tpp.Normal(self.m_a, self.log_s_a.exp()))

        mean_b = self.w_b * tr['a'] + self.b_b
        tr.sample('b', tpp.Normal(mean_b, self.log_s_b.exp()))

        mean_c = self.w_c * tr['b'] + self.b_c
        tr.sample('c', tpp.Normal(mean_c, self.log_s_c.exp()))

        mean_d = self.w_d * tr['c'] + self.b_d
        tr.sample('d', tpp.Normal(mean_d, self.log_s_d.exp()))




data = tpp.sample(P, sizes)

a = []
bs = []
cs = []
ds = []
obss = []
for i in range(1000):
    sample = tpp.sample(P, sizes)

model = tpp.Model(P, lambda tr: None, {'obs': data['obs']})

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=10
print("K={}".format(K))
for i in range(20000):
    opt.zero_grad()
    elbo = model.elbo(K)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())
