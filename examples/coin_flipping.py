### Coin flipping Example


import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from torch.distributions import transforms
from torchdim import dims

device = t.device("cuda" if t.cuda.is_available() else "cpu")
### data
y = {'obs': t.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])}


N = 10
plate_1 = dims(1 , [N])
def P(tr):
    # define the hyperparameters that control the Beta prior
    alpha0 = t.tensor(10.0)
    beta0 = t.tensor(10.0)
    # sample f from the Beta prior
    tr['latent_fairness'] = tpp.Beta(alpha0,beta0)

    tr['obs'] = tpp.Bernoulli(tr['latent_fairness'], sample_dim=plate_1)



class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.alphaq = nn.Parameter(t.tensor(1.2))
        self.betaq = nn.Parameter(t.tensor(1.2))

    def forward(self, tr):
        tr['latent_fairness'] = tpp.Beta(self.alphaq.exp(), self.betaq.exp())



model = tpp.Model(P, Q(), y)
model.to(device)

opt = t.optim.Adam(model.parameters(), lr=1e-3)

K = 10
Ks = tpp.make_dims(P, K)
print("K={}".format(K))
for i in range(15000):
    opt.zero_grad()
    elbo = model.elbo(dims=Ks)
    (-elbo).backward()
    opt.step()

    if 0 == i%100:
        print(elbo.item())



# grab the learned variational parameters
alpha_q = model.Q.alphaq.exp().item()
beta_q = model.Q.betaq.exp().item()

# here we use some facts about the Beta distribution
# compute the inferred mean of the coin's fairness
inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * factor**(1/2)

print("\nBased on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))
