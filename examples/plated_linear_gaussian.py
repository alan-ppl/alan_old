import torch as t
import torch.nn as nn
import alan
t.manual_seed(0)

J = 2
M = 3
N = 4
platesizes = {'plate_1': J, 'plate_2': M, 'plate_3': N}
def P(tr):
    tr.sample('a',   alan.Normal(t.zeros(()), 1))
    tr.sample('b',   alan.Normal(tr['a'], 1))
    tr.sample('c',   alan.Normal(tr['b'], 1), plates='plate_1')
    tr.sample('d',   alan.Normal(tr['c'], 1), plates='plate_2')
    tr.sample('obs', alan.Normal(tr['d'], 1), plates='plate_3')

#def Q(tr):
#    tr.sample('a',   alan.Normal(t.zeros(()), 1))
#    tr.sample('b',   alan.Normal(tr['a'], 1))
#    tr.sample('c',   alan.Normal(tr['b'], 1), plate='plate_1')
#    tr.sample('d',   alan.Normal(tr['c'], 1), plate='plate_2')

class Q(alan.QModule):
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
        tr.sample('a', alan.Normal(self.m_a, self.log_s_a.exp()))

        mean_b = self.w_b * tr['a'] + self.b_b
        tr.sample('b', alan.Normal(mean_b, self.log_s_b.exp()))

        mean_c = self.w_c * tr['b'] + self.b_c
        tr.sample('c', alan.Normal(mean_c, self.log_s_c.exp()))

        mean_d = self.w_d * tr['c'] + self.b_d
        tr.sample('d', alan.Normal(mean_d, self.log_s_d.exp()))




data = alan.sample(P, platesizes)

a = []
bs = []
cs = []
ds = []
obss = []
for i in range(1000):
    sample = alan.sample(P, platesizes)

#model = alan.Model(P, lambda tr: None, {'obs': data['obs']})
model = alan.Model(P, Q(), {'obs': data['obs']})

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
