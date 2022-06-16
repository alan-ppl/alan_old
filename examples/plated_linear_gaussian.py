import torch as t
import torch.nn as nn
import tpp


N = 2
M = 2
K = 2
def P(tr):
    tr['a'] = tpp.Normal(t.zeros(()), 1)
    tr['b'] = tpp.Normal(tr['a'], 1)
    tr['c'] = tpp.Normal(tr['b'], 1, sample_shape=K, sample_names='plate_1')
    tr['d'] = tpp.Normal(tr['c'], 1, sample_shape=M, sample_names='plate_2')
    tr['obs'] = tpp.Normal(tr['d'], 1, sample_shape=N, sample_names='plate_3')





class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_a = nn.Parameter(t.zeros((1)))

        self.w_b = nn.Parameter(t.zeros((1)))
        self.b_b = nn.Parameter(t.zeros((1)))

        self.w_c = nn.Parameter(t.zeros((K,1)))
        self.b_c = nn.Parameter(t.zeros((K,), names=('plate_1',)))

        self.w_d = nn.Parameter(t.zeros((M, K, K)))
        self.b_d = nn.Parameter(t.zeros((M, K), names=('plate_2','plate_1')))

        self.log_s_a = nn.Parameter(t.zeros((1)))
        self.log_s_b = nn.Parameter(t.zeros((1)))
        self.log_s_c = nn.Parameter(t.zeros((K,), names=('plate_1',)))
        self.log_s_d = nn.Parameter(t.zeros((M, K), names=('plate_2','plate_1')))

    def forward(self, tr):
        tr['a'] = tpp.Normal(self.m_a, self.log_s_a.exp())

        if 'K' in tr['a'].names:
            a = tr['a'].mean('K')
        else:
            a = tr['a']
        mean_b = self.w_b @ a + self.b_b
        tr['b'] = tpp.Normal(mean_b, self.log_s_b.exp())

        if 'K' in tr['b'].names:
            b = tr['a'].mean('K')
        else:
            b = tr['a']
        mean_c = self.w_c @ b + self.b_c
        tr['c'] = tpp.Normal(mean_c, self.log_s_c.exp())

        if 'K' in tr['c'].names:
            c = tr['c'].mean('K')
        else:
            c = tr['c']
        mean_d = self.w_d @ c + self.b_d

        tr['d'] = tpp.Normal(mean_d, self.log_s_d.exp())


data = tpp.sample(P)

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
    ds.append(sample['d'].align_to('plate_2', 'plate_1').rename(None).T.flatten())
    obss.append(sample['obs'].rename(None).flatten())



A = t.hstack([t.zeros(K*M*N,K+2), t.vstack([t.eye(K*M)]*N)])

params_prior_cov = t.round(t.cov(t.cat((t.vstack(a),t.vstack(bs),t.vstack(cs), t.vstack(ds)), dim=1).T))
cov_obs_prior = t.eye(K*M*N)

post_cov_inv = (t.inverse(params_prior_cov) + A.T @ t.inverse(cov_obs_prior) @ A)
post_mean = t.inverse(post_cov_inv) @ (A.T @ t.inverse(cov_obs_prior) @  data['obs'].rename(None).flatten())



model = tpp.Model(P, Q(), {'obs': data['obs']})

opt = t.optim.Adam(model.parameters(), lr=1E-3)

print("K=10")
for i in range(5000):
    opt.zero_grad()
    elbo = model.elbo(K=10)
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
for i in range(10000):
    sample = tpp.sample(model.Q)
    a.append(sample['a'].rename(None).flatten())
    bs.append(sample['b'].rename(None).flatten())
    cs.append(sample['c'].rename(None).flatten())
    ds.append(sample['d'].align_to('plate_2', 'plate_1').rename(None).T.flatten())

# Maybe consider saving samples from P and Q and calculating covariance matrix directly to take advantage of plating being matched (i.e
# before flattening)
params_post_cov = t.round(t.cov(t.cat((t.vstack(a),t.vstack(bs),t.vstack(cs), t.vstack(ds)), dim=1).T), decimals=3)
params_post_mean = t.mean(t.cat((t.vstack(a),t.vstack(bs),t.vstack(cs), t.vstack(ds)), dim=1).T, dim=1)

print('Exact posterior Covariance')
print(t.round(t.inverse(post_cov_inv), decimals=4))
print('Approximate Covariance')
print(params_post_cov)

print('Exact posterior Mean')
print(post_mean)
print('Approximate Mean')
print(params_post_mean)
