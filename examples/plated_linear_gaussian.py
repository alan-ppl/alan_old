import torch as t
import torch.nn as nn
import tpp
from tpp.cartesian_tensor import CartesianTensor
J = 2
M = 3
N = 4
def P(tr):
    tr['a'] = tpp.Normal(t.zeros(()), 1)
    tr['b'] = tpp.Normal(tr['a'], 1)
    tr['c'] = tpp.Normal(tr['b'], 1, sample_shape=J, sample_names='plate_1')
    tr['d'] = tpp.Normal(tr['c'], 1, sample_shape=M, sample_names='plate_2')
    tr['obs'] = tpp.Normal(tr['d'], 1, sample_shape=N, sample_names='plate_3')





class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_a = nn.Parameter(t.zeros(()))

        self.w_b = nn.Parameter(t.zeros(()))
        self.b_b = nn.Parameter(t.zeros(()))

        self.w_c = nn.Parameter(t.zeros((J,), names = ('plate_1',)))
        self.b_c = nn.Parameter(t.zeros((J,), names=('plate_1',)))

        self.w_d = nn.Parameter(t.zeros((M, J), names=('plate_2','plate_1')))
        self.b_d = nn.Parameter(t.zeros((M, J), names=('plate_2', 'plate_1')))

        self.log_s_a = nn.Parameter(t.zeros(()))
        self.log_s_b = nn.Parameter(t.zeros(()))
        self.log_s_c = nn.Parameter(t.zeros((J,), names=('plate_1',)))
        self.log_s_d = nn.Parameter(t.zeros((M, J), names=('plate_2','plate_1')))


    def forward(self, tr):
        tr['a'] = tpp.Normal(self.m_a, self.log_s_a.exp())

        mean_b = self.w_b * tr['a'] + self.b_b
        tr['b'] = tpp.Normal(mean_b, self.log_s_b.exp())


        mean_c = self.w_c*tr['b'] + self.b_c
        tr['c'] = tpp.Normal(mean_c, self.log_s_c.exp())

        c = tr['c']
        mean_d = self.w_d * tr['c'] + self.b_d

        tr['d'] = tpp.Normal(mean_d, self.log_s_d.exp())


data = tpp.sample(P)

a = []
bs = []
cs = []
ds = []
obss = []
for i in range(10000):
    sample = tpp.sample(P)
    a.append(sample['a'].rename(None).flatten())
    bs.append(sample['b'].rename(None).flatten())
    cs.append(sample['c'].rename(None).flatten())
    ds.append(sample['d'].align_to('plate_2', 'plate_1').rename(None).T.flatten())
    obss.append(sample['obs'].rename(None).flatten())



A = t.hstack([t.zeros(J*M*N,J+2), t.vstack([t.eye(J*M)]*N)])

params_prior_cov = t.round(t.cov(t.cat((t.vstack(a),t.vstack(bs),t.vstack(cs), t.vstack(ds)), dim=1).T))
cov_obs_prior = t.eye(J*M*N)

post_cov_inv = (t.inverse(params_prior_cov) + A.T @ t.inverse(cov_obs_prior) @ A)
post_mean = t.inverse(post_cov_inv) @ (A.T @ t.inverse(cov_obs_prior) @  data['obs'].rename(None).flatten())



model = tpp.Model(P, Q(), {'obs': data['obs']})

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K = 3
print("K={}".format(K))
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(K=K)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())


post_cov_inv = (t.inverse(params_prior_cov ) + A.T @ t.inverse(cov_obs_prior) @ A)

post_mean = t.inverse(post_cov_inv) @ (A.T @ t.inverse(cov_obs_prior) @  data['obs'].rename(None).flatten())



a = []
bs = []
cs = []
ds = []
for i in range(5000):
    sample = tpp.sample(model.Q)
    # print(sample)
    a.append(sample['a'].rename(None).flatten())
    bs.append(sample['b'].rename(None).flatten())
    cs.append(sample['c'].rename(None).flatten())
    ds.append(sample['d'].align_to('plate_2', 'plate_1').rename(None).T.flatten())



params_post_cov = t.round(t.cov(t.cat((t.vstack(a),t.vstack(bs),t.vstack(cs), t.vstack(ds)), dim=1).T), decimals=2)
params_post_mean = t.mean(t.cat((t.vstack(a),t.vstack(bs),t.vstack(cs), t.vstack(ds)), dim=1).T, dim=1)



print('Exact posterior Covariance')
print(t.round(t.inverse(post_cov_inv), decimals=4))
print('Approximate Covariance')
print(params_post_cov)

print('Exact posterior Mean')
print(post_mean)
print('Approximate mean')
print(params_post_mean)


# print(model.Q.log_s_a.exp())
# print(model.Q.log_s_b.exp())
# print(model.Q.log_s_c.exp())
# print(model.Q.log_s_d.exp())