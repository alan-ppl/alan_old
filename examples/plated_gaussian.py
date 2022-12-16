import torch as t
import torch.nn as nn
import tpp

N = 10
sizes = {'plate_1':N}
def P(tr):
    '''
    Bayesian Gaussian Model
    '''
    a = t.zeros(5,)
    tr.sample('mu',   tpp.MultivariateNormal(a, t.eye(5)))
    tr.sample('obs',   tpp.Normal(tr['mu'], 1), plate='plate_1')



class Q(tpp.Q):
    def __init__(self):
        super().__init__()
        self.reg_param("m_mu", t.zeros(5,))
        self.reg_param("log_s_mu", t.zeros(5,))

    def forward(self, tr):
        tr.sample('mu',   tpp.MultivariateNormal(self.m_mu, t.diag(self.log_s_mu.exp())))

data = tpp.sample(P, sizes)
model = tpp.Model(P, Q(), {'obs': data['obs']})

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K = 3
print("K={}".format(K))
for i in range(15000):
    opt.zero_grad()
    elbo = model.elbo(K=K)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())


print("Approximate mu")
print(model.Q.m_mu)

print("Approximate Covariance")
print(model.Q.log_s_mu.exp())


data_obs = data['obs']

b_n = t.mm(t.inverse(t.eye(5) + 1/N * t.eye(5)),data_obs.mean(axis=0).reshape(-1,1))
A_n = t.inverse(t.eye(5) + 1/N * t.eye(5)) * 1/N

print("True mu")
print(b_n)

print("True covariance")
print(t.diag(A_n))

print((t.abs(b_n - model.Q.m_mu.reshape(-1,1))<0.1).all())
print((t.abs(A_n - t.diag(model.Q.log_s_mu.exp()))<0.1).all())
