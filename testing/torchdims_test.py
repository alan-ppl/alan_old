import torch as t
from torchdim import dims
import tpp

i, j, K = dims(3)
K.size = 5

A = t.ones(3,1)
Ad = A[i,j]

B = t.ones(5,3)
BK = B[K,:]
AdK_sample = tpp.Normal(Ad, 1).rsample(K)
print(AdK_sample)
print(tpp.Normal(Ad, 1).log_prob(AdK_sample))


Ad_sample = tpp.Normal(Ad, 1).rsample()

print(Ad_sample)
print(tpp.Normal(Ad, 1).log_prob(Ad_sample))


A_sample = tpp.Normal(A, 1).rsample()

print(A_sample)
print(tpp.Normal(A, 1).log_prob(A_sample))

BK_sample = tpp.Normal(BK, 1).rsample()

print(BK_sample)
print(tpp.Normal(BK, 1).log_prob(BK_sample))
