import torch as t
import torch.nn as nn
import alan
t.manual_seed(0)

J = 2
M = 3
N = 4
platesizes = {'plate_1': J, 'plate_2': M, 'plate_3': N}

class PQ(alan.PQ):
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
        tr.Q('a', alan.Normal(self.m_a, self.log_s_a.exp()))

        mean_b = self.w_b * tr['a'] + self.b_b
        tr.Q('b', alan.Normal(mean_b, self.log_s_b.exp()))

        mean_c = self.w_c * tr['b'] + self.b_c
        tr.Q('c', alan.Normal(mean_c, self.log_s_c.exp()))

        mean_d = self.w_d * tr['c'] + self.b_d
        tr.Q('d', alan.Normal(mean_d, self.log_s_d.exp()))

        tr.P('a',   alan.Normal(0., 1.))
        tr.P('b',   alan.Normal(tr['a'], 1))
        tr.P('c',   alan.Normal(tr['b'], 1), plates='plate_1')
        tr.P('d',   alan.Normal(tr['c'], 1), plates='plate_2')
        tr.P('obs',  alan.Normal(tr['d'], 1), plates='plate_3')

def pq(tr):
    tr.PQ('a',   alan.Normal(0., 1.))
    tr.PQ('b',   alan.Normal(tr['a'], 1))
    tr.PQ('c',   alan.Normal(tr['b'], 1), plates='plate_1')
    tr.PQ('d',   alan.Normal(tr['c'], 1), plates='plate_2')
    tr.P('obs',  alan.Normal(tr['d'], 1), plates='plate_3')


data = alan.sample_P(pq, platesizes=platesizes, varnames=('/obs',))
model = alan.Model(pq, data=data)
print(model.SampleMP(K=1000, reparam=True).elbo())

model = alan.Model(PQ(), data=data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=10
print("K={}".format(K))
for i in range(20000):
    opt.zero_grad()
    elbo = model.SampleMP(K=K, reparam=True).elbo()
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())
