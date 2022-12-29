### Coin flipping Example
import torch as t
import torch.nn as nn
import alan
from torch.distributions import transforms




N = 10
sizes = {'plate_1':N}
def P(tr):
    # define the hyperparameters that control the Beta prior
    alpha0 = t.tensor(10.0)
    beta0 = t.tensor(10.0)
    tr.sample('latent_fairness', alan.Beta(alpha0, beta0))
    tr.sample('obs', alan.Bernoulli(tr['latent_fairness']), plates='plate_1')



class Q(alan.QModule):
    def __init__(self):
        super().__init__()
        self.alphaq = nn.Parameter(t.tensor(1.2))
        self.betaq = nn.Parameter(t.tensor(1.2))

    def forward(self, tr):
        tr.sample('latent_fairness', alan.Beta(self.alphaq, self.betaq))





data = alan.sample(P, sizes)
model = alan.Model(P, Q(), {'obs': data['obs']})
print(data)

opt = t.optim.Adam(model.parameters(), lr=1e-3)

K = 10
print("K={}".format(K))
for i in range(5000):
    opt.zero_grad()
    elbo = model.elbo(K=K)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
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
