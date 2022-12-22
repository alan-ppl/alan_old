import torch as t
import torch.nn as nn
import tpp

t.autograd.set_detect_anomaly(True)
'''
Test posterior inference with a general gaussian
'''
sigma_0 = t.rand(5,5)
sigma_0 = sigma_0 @ sigma_0.mT
sigma_0 = sigma_0 + t.eye(5) * 1e-5
sigma = t.rand(5,5)
sigma = sigma @ sigma.mT
sigma = sigma + t.eye(5)* 1e-5
a = t.randn(5,)

N = 10
sizes = {'plate_1': N}
def P(tr):
    '''
    Bayesian Gaussian Model
    '''
    tr.sample('mu',   tpp.MultivariateNormal(a, sigma_0))
    tr.sample('obs',   tpp.MultivariateNormal(tr['mu'], sigma), plates='plate_1')




class Q(tpp.Q):
    def __init__(self):
        super().__init__()
        self.reg_param("m_mu", t.zeros(5,))
        self.reg_param("s_mu", t.eye(5))

    def forward(self, tr):
        sigma_nn = self.s_mu @ self.s_mu.mT
        sigma_nn = sigma_nn + t.eye(5) * 1e-5

        tr.sample('mu',   tpp.MultivariateNormal(self.m_mu, sigma_nn))

data = tpp.sample(P, sizes)
model = tpp.Model(P, Q(), {'obs': data['obs']})

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=5
print("K={}".format(K))

for i in range(20000):
    opt.zero_grad()
    elbo = model.elbo(K)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())

inferred_mean = model.Q.m_mu
inferred_cov = t.mm(model.Q.s_mu, model.Q.s_mu.t())
inferred_cov.add_(t.eye(5)* 1e-5)


y_hat = data['obs'].mean(0).reshape(-1,1)

true_cov = t.inverse(N * t.inverse(sigma) + t.inverse(sigma_0))
true_mean = (true_cov @ (N*t.inverse(sigma) @ y_hat + t.inverse(sigma_0)@a.reshape(-1,1))).reshape(1,-1)


print('True Covariance')
print(true_cov)

print('Inferred Covariance')
print(inferred_cov)

print('True Mean')
print(true_mean)

print('Inferred Mean')
print(inferred_mean)
