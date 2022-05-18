'''
Test posterior inference with a general gaussian
'''
sigma_0 = t.rand(5,5)
sigma_0 = t.mm(sigma_0, sigma_0.t())
sigma_0.add_(t.eye(5) * 1e-5)
sigma = t.rand(5,5)
sigma = t.mm(sigma, sigma.t())
sigma.add_(t.eye(5)* 1e-5)
a = t.randn(5,)
def P(tr):
  '''
  Bayesian Heirarchical Gaussian Model
  '''

  tr['mu'] = tpp.MultivariateNormal(a, sigma_0)
  tr['obs'] = tpp.MultivariateNormal(tr['mu'], sigma, sample_shape=10, sample_names='plate_1')



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(5,))

        self.log_s_mu = nn.Parameter(t.randn(5,5))

    def forward(self, tr):
        sigma_nn = t.mm(self.log_s_mu, self.log_s_mu.t())
        sigma_nn.add_(t.eye(5) * 1e-5)
        tr['mu'] = tpp.MultivariateNormal(self.m_mu, covariance_matrix=sigma_nn)

data = tpp.sample(P, "obs")

model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

for i in range(30000):
    opt.zero_grad()
    elbo = model.elbo(K=100)
    (-elbo).backward()
    opt.step()


inferred_mean = model.Q.m_mu
inferred_cov = t.mm(model.Q.log_s_mu, model.Q.log_s_mu.t())
inferred_cov.add_(t.eye(5)* 1e-5)


true_mean = t.mm(sigma_0,t.mm(t.inverse(sigma_0 + 1/10 * sigma),data['obs'].rename(None).mean(axis=0).reshape(-1,1))) + 1/10 * t.mm(sigma,t.mm(t.inverse(sigma_0 + 1/10*sigma),a.reshape(-1,1)))
true_cov = t.mm(t.mm(sigma_0,t.inverse(sigma_0 + 1/10*sigma)),sigma)


assert((t.abs(true_mean - inferred_mean.reshape(-1,1))<0.1).all())
assert(((inferred_cov-true_cov)<0).all())
