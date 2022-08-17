import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
import math
from torchdim import dims

N = 128
plate_1 = dims(1 , [N])
def P(tr):
  tr['theta'] = tpp.Normal(t.zeros(()),1)
  tr['z'] = tpp.Normal(tr['theta'], 1, sample_dim=plate_1)
  tr['obs'] = tpp.Normal(tr['z'], 1)



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_theta = nn.Parameter(t.zeros(1,))
        self.log_s_theta = nn.Parameter(t.zeros(1,))

        self.m_z = nn.Parameter(t.zeros(1,))
        self.log_s_z = nn.Parameter(math.log(2) * t.ones(1,))


    def forward(self, tr):
        tr['theta'] = tpp.Normal(self.m_theta, self.log_s_theta.exp())
        tr['z'] = tpp.Normal(self.m_z, self.log_s_z.exp(), sample_dim=plate_1)

data = tpp.sample(P, "obs")


print(data)
model = tpp.Model(P, Q(), data)

opt = t.optim.Adam(model.parameters(), lr=1E-3)

K=128
dims = tpp.make_dims(P, K)
print("K={}".format(K))
for i in range(10000):
    opt.zero_grad()
    elbo = model.elbo(dims=dims)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())


# print("Approximate mu")
# print(model.Q.m_mu)
#
# print("Approximate Covariance")
# print(model.Q.log_s_mu.exp())
#
# b_n = t.mm(t.inverse(t.eye(5) + t.eye(5)),data['obs'].rename(None).reshape(-1,1))
# A_n = t.inverse(t.eye(5) + t.eye(5))
#
# print("True mu")
# print(b_n)
#
# print("True covariance")
# print(t.diag(A_n))
