import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from functorch.dim import dims

J = 2
M = 3
N = 4
plate_1,plate_2,plate_3 = dims(3 , [J, M, N])
def P(tr):
    tr['a'] = tpp.Normal(t.zeros(()), 1)
    tr['b'] = tpp.Normal(tr['a'], 1)
    tr['c'] = tpp.Normal(tr['b'], 1, sample_dim=plate_1)
    tr['d'] = tpp.Normal(tr['c'], 1, sample_dim=plate_2)
    tr['obs'] = tpp.Normal(tr['d'], 1, sample_dim=plate_3)


class Q(tpp.Q_module):
    def __init__(self):
        super().__init__()
        self.reg_param("m_a", t.zeros(()))
        self.reg_param("w_b", t.zeros(()))
        self.reg_param("b_b", t.zeros(()))

        self.reg_param("w_c", t.zeros((J,)), [plate_1])
        self.reg_param("b_c", t.zeros((J,)), [plate_1])

        self.reg_param("w_d", t.zeros((M, J)), [plate_2,plate_1])
        self.reg_param("b_d", t.zeros((M, J)), [plate_2,plate_1])

        self.reg_param("log_s_a", t.zeros(()))
        self.reg_param("log_s_b", t.zeros(()))
        self.reg_param("log_s_c", t.zeros((J,)), [plate_1])
        self.reg_param("log_s_d", t.zeros((M,J)), [plate_2,plate_1])


    def forward(self, tr):
        tr['a'] = tpp.Normal(self.m_a, self.log_s_a.exp())
        mean_b = self.w_b * tr['a'] + self.b_b
        tr['b'] = tpp.Normal(mean_b, self.log_s_b.exp())

        mean_c = self.w_c * tr['b'] + self.b_c

        tr['c'] = tpp.Normal(mean_c, self.log_s_c.exp())

        mean_d = self.w_d * tr['c'] + self.b_d
        tr['d'] = tpp.Normal(mean_d, self.log_s_d.exp())




data = tpp.sample(P)

a = []
bs = []
cs = []
ds = []
obss = []
for i in range(1000):
    sample = tpp.sample(P)
    a.append(tpp.dename(sample['a']).flatten())
    bs.append(tpp.dename(sample['b']).flatten())
    cs.append(tpp.dename(sample['c']).flatten())
    ds.append(tpp.dename(sample['d']).flatten())
    obss.append(tpp.dename(sample['obs']).flatten())



A = t.hstack([t.zeros(J*M*N,J+2), t.vstack([t.eye(J*M)]*N)])

params_prior_cov = t.round(t.cov(t.cat((t.vstack(a),t.vstack(bs),t.vstack(cs), t.vstack(ds)), dim=1).T))
cov_obs_prior = t.eye(J*M*N)

post_cov_inv = (t.inverse(params_prior_cov) + A.T @ t.inverse(cov_obs_prior) @ A)
post_mean = t.inverse(post_cov_inv) @ (A.T @ t.inverse(cov_obs_prior) @  tpp.dename(data['obs']).flatten())



model = tpp.Model(P, Q(), {'obs': data['obs']})

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=2
dims = tpp.make_dims(P, K, [plate_1,plate_2,plate_3])
print("K={}".format(K))
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(dims=dims)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())


post_cov_inv = (t.inverse(params_prior_cov ) + A.T @ t.inverse(cov_obs_prior) @ A)

post_mean = t.inverse(post_cov_inv) @ (A.T @ t.inverse(cov_obs_prior) @  tpp.dename(data['obs']).flatten())



a = []
bs = []
cs = []
ds = []
for i in range(5000):
    sample = tpp.sample(model.Q)
    # print(sample)
    a.append(tpp.dename(sample['a']).flatten())
    bs.append(tpp.dename(sample['b']).flatten())
    cs.append(tpp.dename(sample['c']).flatten())
    ds.append(tpp.dename(sample['d']).flatten())


params_post_cov = t.round(t.cov(t.cat((t.vstack(a),t.vstack(bs),t.vstack(cs), t.vstack(ds)), dim=1).T), decimals=4)
params_post_mean = t.mean(t.cat((t.vstack(a),t.vstack(bs),t.vstack(cs), t.vstack(ds)), dim=1).T, dim=1)



print('Exact posterior Covariance')
print(t.round(t.inverse(post_cov_inv), decimals=4))
print('Approximate Covariance')
print(params_post_cov)

print('Exact posterior Mean')
print(post_mean)
print('Approximate mean')
print(params_post_mean)
