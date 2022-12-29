import torch as t
from functorch.dim import dims
import alan

i, j, K = dims(3)
K.size = 5

A = t.ones(3,1)
Ad = A[i,j]

B = t.ones(5,3)
BK = B[K,:]
AdK_sample = alan.Normal(Ad, 1).rsample(K)
print(AdK_sample)
print(alan.Normal(Ad, 1).log_prob(AdK_sample))


Ad_sample = alan.Normal(Ad, 1).rsample()

print(Ad_sample)
print(alan.Normal(Ad, 1).log_prob(Ad_sample))


A_sample = alan.Normal(A, 1).rsample()

print(A_sample)
print(alan.Normal(A, 1).log_prob(A_sample))

BK_sample = alan.Normal(BK, 1).rsample()

print(BK_sample)
print(alan.Normal(BK, 1).log_prob(BK_sample))
