import torch as t
import torch.nn as nn
import tpp


N = 2
def P(tr):
    tr['a'] = tpp.Normal(t.zeros(()), 1)
    tr['b'] = tpp.Normal(tr['a'], 1)
    tr['c'] = tpp.Normal(tr['b'], 1, sample_shape=2, sample_names='plate_1')
    tr['d'] = tpp.Normal(tr['c'], 1, sample_shape=2, sample_names='plate_2')
    tr['obs'] = tpp.Normal(tr['d'], 1, sample_shape=N, sample_names='plate_3')


class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_a = nn.Parameter(t.zeros(()))
        self.m_b = nn.Parameter(t.zeros(()))
        self.m_c = nn.Parameter(t.zeros((2,), names=('plate_1',)))
        self.m_d = nn.Parameter(t.zeros((2, 2), names=('plate_2','plate_1')))

        self.log_s_a = nn.Parameter(t.zeros(()))
        self.log_s_b = nn.Parameter(t.zeros(()))
        self.log_s_c = nn.Parameter(t.zeros((2,), names=('plate_1',)))
        self.log_s_d = nn.Parameter(t.zeros((2, 2), names=('plate_2','plate_1')))

    def forward(self, tr):
        tr['a'] = tpp.Normal(self.m_a, self.log_s_a.exp())
        tr['b'] = tpp.Normal(self.m_b, self.log_s_b.exp())
        tr['c'] = tpp.Normal(self.m_c, self.log_s_c.exp())
        tr['d'] = tpp.Normal(self.m_d, self.log_s_d.exp())
        # print(tr['a'].shape)
        # print(tr['a'].names)
        # print(tr['b'].shape)
        # print(tr['b'].names)
        # print(tr['c'].shape)
        # print(tr['c'].names)
        # print(tr['d'].shape)

data = tpp.sample(P)


# trq = tpp.TraceSampleLogQ(K=10, data={'obs': data['obs']})
# Q = Q()
#
# Q(trq)
# trp = tpp.TraceLogP(trq.sample, {'obs': data['obs']})
# P(trp)

a = []
bs = []
cs = []
ds = []
obss = []
for i in range(10000):
    a.append(tpp.sample(P)['a'].rename(None).flatten())
    bs.append(tpp.sample(P)['b'].rename(None).flatten())
    cs.append(tpp.sample(P)['c'].rename(None).flatten())
    ds.append(tpp.sample(P)['d'].rename(None).flatten())
    obss.append(tpp.sample(P)['obs'].rename(None).flatten())

a_prior_cov = t.round(t.cov(t.stack(a).T))
a_prior_mean = t.round(t.mean(t.stack(a).T))
# print('a_prior_cov')
# print(a_prior_cov)
# print('a_prior_mean')
# print(a_prior_mean)
b_prior_cov = t.round(t.cov(t.stack(bs).T))
b_prior_mean = t.round(t.mean(t.stack(bs).T))
# print('b_prior_cov')
# print(b_prior_cov)
# print('b_prior_mean')
# print(b_prior_mean)
c_prior_cov = t.round(t.cov(t.stack(cs).T))
c_prior_mean = t.round(t.mean(t.stack(cs).T, dim = 1))
print('c_prior_cov')
print(c_prior_cov)
print('c_prior_mean')
print(c_prior_mean)

print('a,b cov')
print(t.round(t.cov(t.cat((t.stack(a).T,t.stack(bs).T)))))

print('a,b,c cov')
print(t.round(t.cov(t.cat((t.stack(a).T,t.stack(bs).T,t.stack(cs).T)))))

print('a,b,c,d cov')
print(t.round(t.cov(t.cat((t.stack(a).T,t.stack(bs).T,t.stack(cs).T, t.stack(ds).T)))))

d_prior_cov = t.round(t.cov(t.stack(ds).T))
d_prior_mean = t.round(t.mean(t.stack(ds).T, dim = 1).reshape(2,2))
print('d_prior_cov')
print(d_prior_cov)
print('d_prior_mean')
print(d_prior_mean)
obs_prior_cov = t.round(t.cov(t.stack(obss).T))
print(obs_prior_cov)
obs_prior_mean = t.round(t.mean(t.stack(obss).T, dim = 1).reshape(2,2,2))
# print('obs_prior_cov')
# print(obs_prior_cov)
# print('obs_prior_mean')
# print(obs_prior_mean)


print('a obs joint cov ')
print(t.round(t.cov(t.cat((t.stack(a).T,t.stack(obss).T)))))



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


print("Approximate mean a")
print(model.Q.m_a)

print("Approximate variance a")
print(model.Q.log_s_a.exp()**2)

print("Posterior a variance")

w_a = t.ones(N*2*2,1)
#w_a[2::3] = 1
print(w_a.shape)
cov_a = (1/a_prior_cov +  w_a.T @ t.inverse(obs_prior_cov) @ w_a)
print(1/cov_a)

print("Posterior a mean")
mean_a = 1/cov_a * ((w_a.T @ t.inverse(obs_prior_cov) @ data['obs'].rename(None).flatten()))
print(mean_a)


print("Approximate mean b")
print(model.Q.m_b)

print("Approximate variance b")
print(model.Q.log_s_b.exp()**2)

print("Posterior b variance")
w_b = t.ones(8 ,1)
cov_b = (1/b_prior_cov +  w_b.T @ t.inverse(obs_prior_cov) @ w_b)
print(1/cov_b)

print("Posterior b mean")
mean_b = 1/cov_b * ((w_b.T @ t.inverse(obs_prior_cov) @  data['obs'].rename(None).flatten()))
print(mean_b)


print("Approximate mean c")
print(model.Q.m_c)

print("Approximate variance c")
print(model.Q.log_s_c.rename(None).exp()**2)

print("Posterior c variance")
w_c = t.ones(8 ,2)
cov_c = (t.inverse(c_prior_cov) + w_c.T @ t.inverse(obs_prior_cov) @ w_c)
print(t.inverse(cov_c))

print("Posterior c mean")
mean_c = t.inverse(cov_c) * (w_c.T @ t.inverse(obs_prior_cov) @  data['obs'].rename(None).flatten())
print(mean_c)


print("Approximate mean d")
print(model.Q.m_d)
print(model.Q.m_d.shape)

print("Approximate variance d")
print(model.Q.log_s_d.exp()**2)

print("Posterior d variance")
w_d = t.ones(8 ,4)
cov_d = (t.inverse(d_prior_cov) + w_d.T @ t.inverse(obs_prior_cov) @ w_d)
print(t.inverse(cov_d))

print("Posterior d mean")
print(cov_d.shape)
print(data['obs'].rename(None).mean(axis=0).shape)
mean_d = t.inverse(cov_d) * (w_d.T @ t.inverse(obs_prior_cov) @ data['obs'].rename(None).flatten())
print(mean_d)

# logps = {rv: tpp.backend.sum_none_dims(lp) for (rv, lp) in trp.log_prob().items()}
#
# lp, marginals = tpp.backend.sum_lps(logps.values())
# print("Marginals")
#
#
# print(tpp.backend.gibbs(marginals))
