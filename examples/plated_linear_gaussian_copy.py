import torch as t
import torch.nn as nn
import tpp


N = 1
def P(tr):
    # tr['a'] = tpp.Normal(t.zeros(()), 1)
    # tr['b'] = tpp.Normal(tr['a'], 1)
    # tr['c'] = tpp.Normal(tr['b'], 1, sample_shape=2, sample_names='plate_1')
    # tr['d'] = tpp.Normal(tr['c'], 1, sample_shape=2, sample_names='plate_2')
    # tr['obs'] = tpp.Normal(tr['d'], 1, sample_shape=N, sample_names='plate_3')
    tr['a'] = tpp.Normal(t.zeros((1,1)), 1)
    tr['b'] = tpp.Normal(tr['a'], 1)
    tr['c'] = tpp.Normal(tr['b'] @ t.ones(1,2) , 1)
    tr['d'] = tpp.Normal(tr['c'] @ t.hstack([t.eye(2),t.eye(2)]), 1)
    # print(tr['d'].shape)
    tr['obs'] = tpp.Normal(tr['d'] @ t.hstack([t.eye(4),t.eye(4)]), 1)
    # print(tr['a'].shape)
    # print(tr['b'].shape)
    #print(tr['c'].shape)
    #print(tr['d'].shape)
    # print(tr['obs'].shape)



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_a = nn.Parameter(t.zeros((1)))
        self.m_b = nn.Parameter(t.zeros((1)))
        self.m_c = nn.Parameter(t.zeros((2,)))
        self.m_d = nn.Parameter(t.zeros((4,)))

        self.s_a = nn.Parameter(t.randn(1,1))
        self.s_b = nn.Parameter(t.randn(1,1))
        self.s_c = nn.Parameter(t.randn(2,2))
        self.s_d = nn.Parameter(t.randn(4,4))

    def forward(self, tr):

        sigma_a = self.s_a @ self.s_a.t()
        sigma_a.add_(t.eye(1) * 1e-5)
        tr['a'] = tpp.MultivariateNormal(self.m_a, sigma_a)

        sigma_b = self.s_b @ self.s_b.t()
        sigma_b.add_(t.eye(1) * 1e-5)
        tr['b'] = tpp.MultivariateNormal(self.m_b, sigma_b)

        sigma_c = self.s_c @ self.s_c.t()
        sigma_c.add_(t.eye(2) * 1e-5)
        tr['c'] = tpp.MultivariateNormal(self.m_c, sigma_c)

        sigma_d = self.s_d @ self.s_d.t()
        sigma_d.add_(t.eye(4) * 1e-5)#.refine_names('plate_2', 'plate_1')
        tr['d'] = tpp.MultivariateNormal(self.m_d, sigma_d)


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
for i in range(1000):
    sample = tpp.sample(P)
    a.append(sample['a'].rename(None).flatten())
    bs.append(sample['b'].rename(None).flatten())
    cs.append(sample['c'].rename(None).flatten())
    ds.append(sample['d'].rename(None).flatten())
    obss.append(sample['obs'].rename(None).flatten())

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
# print('c_prior_cov')
# print(c_prior_cov)
# print('c_prior_mean')
# print(c_prior_mean)
#
# print('a,b cov')
# print(t.round(t.cov(t.cat((t.stack(a).T,t.stack(bs).T)))))
#
# print('a,b,c cov')
# print(t.round(t.cov(t.cat((t.stack(a).T,t.stack(bs).T,t.stack(cs).T)))))
#
# print('a,b,c,d cov')
# print(t.round(t.cov(t.cat((t.stack(a).T,t.stack(bs).T,t.stack(cs).T, t.stack(ds).T)))))

d_prior_cov = t.round(t.cov(t.stack(ds).T))
d_prior_mean = t.round(t.mean(t.stack(ds).T, dim = 1).reshape(2,2))
# print('d_prior_cov')
# print(d_prior_cov)
# print('d_prior_mean')
# print(d_prior_mean)
obs_prior_cov = t.round(t.cov(t.stack(obss).T))
# print(obs_prior_cov)
obs_prior_mean = t.round(t.mean(t.stack(obss).T, dim = 1).reshape(2,2,2))
# # print('obs_prior_cov')
# # print(obs_prior_cov)
# # print('obs_prior_mean')
# # print(obs_prior_mean)
#
#
# print('a,b,c,d, obs joint cov ')
# print(t.round(t.cov(t.cat((t.stack(cs).T, t.stack(obss).T)))))

cov_y_by_c = t.round(t.cov(t.stack(obss).T - t.vstack([t.eye(4),t.eye(4)]) @ t.vstack([t.eye(2),t.eye(2)]) @ t.stack(cs).T))
cov_y_by_b = t.round(t.cov(t.stack(obss).T - t.vstack([t.eye(4),t.eye(4)]) @ t.vstack([t.eye(2),t.eye(2)]) @ t.ones(2,1) @ t.stack(bs).T))
cov_y_by_a = t.round(t.cov(t.stack(obss).T - t.vstack([t.eye(4),t.eye(4)]) @ t.vstack([t.eye(2),t.eye(2)]) @ t.ones(2,1) @ t.stack(a).T))

#
# print(cov_y_by_c)
# print(cov_y_by_b)
# print(cov_y_by_a)


model = tpp.Model(P, Q(), {'obs': data['obs']})

opt = t.optim.Adam(model.parameters(), lr=1E-3)

print("K=10")
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(K=1)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())


print("Approximate mean a")
print(model.Q.m_a)

print("True Posterior a mean")
mean_a = 1/cov_a * ((w_a.T @ t.inverse(cov_y_by_a) @ data['obs'].rename(None).flatten()))
print(mean_a)

print("Approximate variance a")
print(model.Q.s_a @ model.Q.s_a.t())

print("Posterior a variance")

w_a = t.vstack([t.eye(4),t.eye(4)]) @ t.vstack([t.eye(2),t.eye(2)]) @ t.ones(2,1)
cov_a = (1/a_prior_cov +  w_a.T @ t.inverse(cov_y_by_a) @ w_a)
print(1/cov_a)




print("Approximate mean b")
print(model.Q.m_b)

print("True Posterior b mean")
mean_b = 1/cov_b * ((w_b.T @ t.inverse(cov_y_by_b) @  data['obs'].rename(None).flatten()))
print(mean_b)

print("Approximate variance b")
print(model.Q.s_b @ model.Q.s_b.t())

print("Posterior b variance")
w_b = t.vstack([t.eye(4),t.eye(4)]) @ t.vstack([t.eye(2),t.eye(2)]) @ t.ones(2,1)
cov_b = (1/b_prior_cov +  w_b.T @ t.inverse(cov_y_by_b) @ w_b)
print(1/cov_b)




print("Approximate mean c")
print(model.Q.m_c)

print("True Posterior c mean")
mean_c = t.inverse(cov_c) @ (w_c.T @ t.inverse(cov_y_by_c) @  data['obs'].rename(None).flatten())
print(mean_c)

print("Approximate variance c")
print(model.Q.s_c @ model.Q.s_c.t())

print("Posterior c variance")
w_c = t.vstack([t.eye(4),t.eye(4)]) @ t.vstack([t.eye(2),t.eye(2)])
cov_c = (t.inverse(c_prior_cov) + w_c.T @ t.inverse(cov_y_by_c) @ w_c)
print(t.inverse(cov_c))




print("Approximate mean d")
print(model.Q.m_d)

print("Posterior d mean")
mean_d = t.inverse(cov_d) @ (w_d.T @ t.inverse(t.eye(8)) @ data['obs'].rename(None).flatten())
print(mean_d)

print("Approximate variance d")
print(model.Q.s_d @ model.Q.s_d.t())

print("Posterior d variance")
w_d = t.vstack([t.eye(4),t.eye(4)])
cov_d = (t.inverse(d_prior_cov) + w_d.T @ t.inverse(t.eye(8)) @ w_d)
print(t.inverse(cov_d))



# logps = {rv: tpp.backend.sum_none_dims(lp) for (rv, lp) in trp.log_prob().items()}
#
# lp, marginals = tpp.backend.sum_lps(logps.values())
# print("Marginals")
#
#
# print(tpp.backend.gibbs(marginals))
