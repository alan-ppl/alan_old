import torch as t
import alan
import math


N = 25
def P(tr):
    tr('ts_1', alan.Normal(0, 1/math.sqrt(N)))
    for i in range(2,N+1):
        tr('ts_{}'.format(i), alan.Normal(tr['ts_{}'.format(i-1)], 1/math.sqrt(N)))
        # print(tr['ts_{}'.format(i)])
    tr('obs', alan.Normal(tr['ts_{}'.format(N)], 1))


model = alan.Model(P)
#data = model.sample_prior(varnames='obs')
data = {'obs': t.tensor(3.)}
cond_model = model.condition(data)

Ks = [3,10,30,100,300]#,1000,3000]
runs = 10
results = t.zeros(len(Ks), runs)
for (i, K) in enumerate(Ks):
    for j in range(runs):
        results[i, j] = cond_model.sample_mp(K).elbo().item()
print(results.mean(-1))

#  
#        
#print(cond_model.sample_global(1).elbo())
#print(cond_model.sample_global(3).elbo())
#print(cond_model.sample_global(10).elbo())
#print(cond_model.sample_global(30).elbo())
#print(cond_model.sample_global(100).elbo())
#print(cond_model.sample_global(300).elbo())
#print(cond_model.sample_global(1000).elbo())
#print(cond_model.sample_global(3000).elbo())
#
#print() 
#print()
#
#print(cond_model.sample_mp(3).elbo())
#print(cond_model.sample_mp(10).elbo())
#print(cond_model.sample_mp(30).elbo())
#print(cond_model.sample_mp(100).elbo())
#print(cond_model.sample_mp(300).elbo())
#print(cond_model.sample_mp(1000).elbo())
#print(cond_model.sample_mp(3000).elbo())

#print(cond_model.sample_tmc(1).elbo())
#print(cond_model.sample_tmc(3).elbo())
#print(cond_model.sample_tmc(10).elbo())
#print(cond_model.sample_tmc(30).elbo())
#print(cond_model.sample_tmc(100).elbo())
#print(cond_model.sample_tmc(300).elbo())
#print(cond_model.sample_tmc(1000).elbo())
#print(cond_model.sample_tmc(3000).elbo())
#
