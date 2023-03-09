import torch as t
import torch.nn as nn
import alan
from alan.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from alan.backend import vi
import tqdm
from functorch.dim import dims

a = t.zeros(5,)
def P(tr):
  '''
  Bayesian Gaussian Model
  '''

  tr['mu'] = alan.Normal(a, t.ones(5,))
  tr['obs'] = alan.Normal(tr['mu'], t.ones(5,))



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_mu = nn.Parameter(t.zeros(5,))

        self.log_s_mu = nn.Parameter(t.zeros(5,))



    def forward(self, tr):
        tr['mu'] = alan.Normal(self.m_mu, self.log_s_mu.exp())

data = alan.sample(P, "obs")
test_data = alan.sample(P, "obs")

print(data)
model = alan.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=5
dims = alan.make_dims(P, K)
print("K={}".format(K))
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(dims=dims)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())


print("Approximate mu")
print(model.Q.m_mu)

print("Approximate Covariance")
print(model.Q.log_s_mu.exp()**2)

b_n = t.mm(t.inverse(t.eye(5) + t.eye(5)),alan.dename(data['obs']).reshape(-1,1))
A_n = t.inverse(t.eye(5) + t.eye(5))

print("True mu")
print(b_n)

print("True covariance")
print(t.diag(A_n))



print('Test Data')
print(test_data)
pred_lik = model.pred_likelihood(test_data=test_data, num_samples=1000).sum()
print(pred_lik)
pred_dist = alan.Normal(model.Q.m_mu, (model.Q.log_s_mu.exp()**2 + t.ones(5,))**(1/2))
true_pred_lik = pred_dist.log_prob(test_data['obs']).sum()
print(true_pred_lik)


assert((t.abs(pred_lik - true_pred_lik)<0.5).all())
