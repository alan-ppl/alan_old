import torch as t
import torch.nn as nn
import tpp

def P(tr): 
    scale = 0.1
    tr['a'] = tpp.Normal(t.zeros(()), 1)
    tr['b'] = tpp.Normal(tr['a'], 1)
    tr['c'] = tpp.Normal(tr['b'], 1, sample_shape=3, sample_names='plate_1')
    tr['d'] = tpp.Normal(tr['c'], 1, sample_shape=4, sample_names='plate_2')
    tr['obs'] = tpp.Normal(tr['d'], 0.1, sample_shape=5, sample_names='plate_3')


class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_a = nn.Parameter(t.zeros(()))
        self.m_b = nn.Parameter(t.zeros(()))
        self.m_c = nn.Parameter(t.zeros((3,), names=('plate_1',)))
        self.m_d = nn.Parameter(t.zeros((4, 3), names=('plate_2','plate_1')))

        self.log_s_a = nn.Parameter(t.zeros(()))
        self.log_s_b = nn.Parameter(t.zeros(()))
        self.log_s_c = nn.Parameter(t.zeros((3,), names=('plate_1',)))
        self.log_s_d = nn.Parameter(t.zeros((4, 3), names=('plate_2','plate_1')))

    def forward(self, tr):
        tr['a'] = tpp.Normal(self.m_a, self.log_s_a.exp())
        tr['b'] = tpp.Normal(self.m_b, self.log_s_b.exp())
        tr['c'] = tpp.Normal(self.m_c, self.log_s_c.exp())
        tr['d'] = tpp.Normal(self.m_d, self.log_s_d.exp())

data = tpp.sample(P, "obs")

model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-2)
# print("K=1")
# for i in range(2000):
#     opt.zero_grad()
#     elbo = model.elbo(K=1)
#     (-elbo).backward()
#     opt.step()

#     if 0 == i%100:
#         print(elbo.item())
    
print("K=10")
for i in range(2000):
    opt.zero_grad()
    elbo = model.elbo(K=10)
    (-elbo).backward()
    opt.step()

    if 0 == i%100:
        print(elbo.item())
    
