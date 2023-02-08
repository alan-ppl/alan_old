import torch as t
import torch.nn as nn
import alan
from alan.postproc import *
t.manual_seed(0)

def P(tr):
  '''
  Bayesian Gaussian Model
  '''
  a = t.zeros(5)
  tr.sample('mu', alan.Normal(a, t.ones(5))) #, plate="plate_1")
  tr.sample('obs', alan.MultivariateNormal(tr['mu'], t.eye(5)))



class Q(alan.QModule):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(5,))
        self.log_s_mu = nn.Parameter(t.zeros(5,))

    def forward(self, tr):
        tr.sample('mu', alan.Normal(self.m_mu, self.log_s_mu.exp())) #, plate="plate_1")

data = alan.sample(P, varnames=('obs','mu')) #, sizes={"plate_1": 2})
print(data)
data = alan.sample(P, varnames=('obs',)) #, sizes={"plate_1": 2})
model = alan.Model(P, Q(), data)


opt = t.optim.Adam(model.parameters(), lr=1E-3)



K=25

# elbos = []
# elbos_tmc = []
# for i in range(100):
#     t.manual_seed(i)
#     elbos.append(model.elbo(K))
#     model = alan.Model(P, Q(), data)
#     elbos_tmc.append(model.elbo_tmc(K))
#
# print(sum(elbos)/100)
# print(sum(elbos_tmc)/100)
print("K={}".format(K))
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo_tmc_new(K)
    # print(elbo)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())



print("Approximate mu")
print(model.Q.m_mu)

print("Approximate Covariance")
print(model.Q.log_s_mu.exp()**2)

b_n = t.mm(t.inverse(t.eye(5) + t.eye(5)),data['obs'].reshape(-1,1))
A_n = t.inverse(t.eye(5) + t.eye(5))

print("True mu")
print(b_n)

print("True covariance")
print(t.diag(A_n))
#
#
# print('Moments')
# print('Mean')
# print(mean(model.weights(10000)))
# print(stderr_mean(model.weights(10000)))
# print('Mean squared')
# print(mean2(model.weights(10000)))
# print(stderr_mean2(model.weights(10000)))
# print('Variance')
# print(var(model.weights(10000)))
# print('Standard Deviation')
# print(std(model.weights(10000)))
# print('Effective Sample Size')
# print(ess(model.weights(10000)))
