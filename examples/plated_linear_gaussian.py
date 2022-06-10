import torch as t
import torch.nn as nn
import tpp

N = 10
def P(tr):
    tr['a'] = tpp.Normal(t.zeros(()), 1)
    tr['b'] = tpp.Normal(tr['a'], 1)
    tr['c'] = tpp.Normal(tr['b'], 1, sample_shape=3, sample_names='plate_1')
    tr['d'] = tpp.Normal(tr['c'], 1, sample_shape=4, sample_names='plate_2')
    tr['obs'] = tpp.Normal(tr['d'], 1, sample_shape=N, sample_names='plate_3')


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

data = tpp.sample(P)


# trq = tpp.TraceSampleLogQ(K=10, data={'obs': data['obs']})
# Q = Q()
#
# Q(trq)
# trp = tpp.TraceLogP(trq.sample, {'obs': data['obs']})
# P(trp)


model = tpp.Model(P, Q(), {'obs': data['obs']})

opt = t.optim.Adam(model.parameters(), lr=1E-3)

print("K=10")
for i in range(1000):
    opt.zero_grad()
    elbo = model.elbo(K=10)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())


print("Approximate mean d")
print(model.Q.m_d)

print("Approximate std d")
print(model.Q.log_s_d.exp()**2)

print("Posterior d std")
cov_d = (1/4 + N * 1/5)
print(1/cov_d)

print("Posterior d mean")
mean_d = 1/cov_d * (1/5 * N * data['obs'].rename(None).mean(axis=0).T)
print(mean_d)

print("Approximate mean c")
print(model.Q.m_c)

print("Approximate std c")
print(model.Q.log_s_c.exp()**2)

print("Posterior c std")
cov_c = (1 + N * 1/5)
print(1/cov_c)

print("Posterior c mean")
mean_c = 1/cov_c * (1/5 * N * data['obs'].rename(None).mean(axis=0).mean(axis=1))
print(mean_c)

print("Approximate mean b")
print(model.Q.m_b)

print("Approximate std b")
print(model.Q.log_s_b.exp()**2)

print("Posterior b std")
cov_b = (1/2 + N * 1/5)
print(1/cov_b)

print("Posterior b mean")
mean_b = 1/cov_b * (1/5 * N * data['obs'].rename(None).mean())
print(mean_b)

print("Approximate mean a")
print(model.Q.m_a)

print("Approximate std a")
print(model.Q.log_s_a.exp()**2)

print("Posterior a std")
cov_a = (1 + N * 1/5)
print(1/cov_a)

print("Posterior a mean")
mean_a = 1/cov_a * (1/5 * N * data['obs'].rename(None).mean())
print(mean_a)

# logps = {rv: tpp.backend.sum_none_dims(lp) for (rv, lp) in trp.log_prob().items()}
#
# lp, marginals = tpp.backend.sum_lps(logps.values())
# print("Marginals")
#
#
# print(tpp.backend.gibbs(marginals))
