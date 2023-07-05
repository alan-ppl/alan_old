import torch as t
import torch.nn as nn
import alan
t.manual_seed(0)

J = 10
M = 20
N = 30
platesizes = {'plate_1': J, 'plate_2': M, 'plate_3': N}
def P(tr):
    tr('a',   alan.Normal(t.zeros(()), 1))
    tr('b',   alan.Normal(tr['a'], 1))
    tr('c',   alan.Normal(tr['b'], 1), plates='plate_1')
    tr('d',   alan.Normal(tr['c'], 1), plates='plate_2')
    tr('obs', alan.Bernoulli(logits=tr['d']), plates='plate_3')

class Q(alan.AlanModule):
    def __init__(self):
        super().__init__()
        self.Na = alan.AMMP_ISNormal()
        self.Nb = alan.AMMP_ISNormal()
        self.Nc = alan.AMMP_ISNormal({'plate_1': J})
        self.Nd = alan.AMMP_ISNormal({'plate_1': J, 'plate_2': M})

    def forward(self, tr):
        tr('a',   self.Na())
        tr('b',   self.Nb())
        tr('c',   self.Nc())
        tr('d',   self.Nd())


data = alan.Model(P).sample_prior(platesizes=platesizes, varnames='obs')

K = 10
T = 1000
lr = 0.2
print('AMMPIS')
t.manual_seed(0)
q = Q()
m1 = alan.Model(P, q).condition(data=data)
for i in range(T):
    sample = m1.sample_same(K, reparam=False)
    m1.ammpis_update(lr, sample)

    if 0 == i%100:
        print(sample.elbo().item())


print('VI')
class Q(alan.AlanModule):
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
        tr('a', alan.Normal(self.m_a, self.log_s_a.exp()))

        mean_b = self.w_b * tr['a'] + self.b_b
        tr('b', alan.Normal(mean_b, self.log_s_b.exp()))

        mean_c = self.w_c * tr['b'] + self.b_c
        tr('c', alan.Normal(mean_c, self.log_s_c.exp()))

        mean_d = self.w_d * tr['c'] + self.b_d
        tr('d', alan.Normal(mean_d, self.log_s_d.exp()))

model = alan.Model(P, Q())

cond_model = alan.Model(P, Q()).condition(data=data)

opt = t.optim.Adam(cond_model.parameters(), lr=1E-3)

K=10
print("K={}".format(K))
for i in range(1000):
    opt.zero_grad()
    elbo = cond_model.sample_cat(K, True).elbo()
    (-elbo).backward()
    opt.step()

    if 0 == i%100:
        print(elbo.item())