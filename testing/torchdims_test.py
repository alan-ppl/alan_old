import torch as t
from torchdim import dims
import tpp

i, j, K = dims(3)
K.size = 5

print(i,j,K)
A = t.ones(3,1)
Ad = A[i,j]
print(Ad)
B = t.ones(5,3)
BK = B[K,:]
AdK_sample = tpp.Normal(Ad, 1).rsample(K)
# Ad_sample = tpp.Normal(Ad, 1).rsample()
# A_sample = tpp.Normal(A, 1).rsample()
# BK_sample = tpp.Normal(BK, 1).rsample()

print(AdK_sample)
# print(tpp.Normal(Ad, 1).log_prob(AdK_sample))

# print(Ad_sample)
# print(tpp.Normal(Ad, 1).log_prob(Ad_sample))
#
# print(A_sample)
# print(tpp.Normal(A, 1).log_prob(A_sample))
#
# print(BK_sample)
# print(tpp.Normal(BK, 1).log_prob(BK_sample))
