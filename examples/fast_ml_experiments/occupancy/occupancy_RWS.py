import torch as t
import torch.nn as nn
import alan
import numpy as np

from alan.experiment_utils import seed_torch

def generate_model(N,M,device,ML=1, run=0, use_data=True):

    M = 6
    J = 12
    I = 150
    Returns = 5
    sizes = {'plate_Years': M, 'plate_Birds':J, 'plate_Ids':I, 'plate_Replicate': 5}



    def P(tr,weather,quality):
      '''
      Hierarchical Occupancy Model
      '''
      tr('year_mean', alan.Normal(tr.zeros(()), tr.ones(())), plates='plate_Years')

      tr('bird_mean', alan.Normal(tr['year_mean'], tr.ones(())), plates='plate_Birds')

      tr('beta', alan.Normal(tr['bird_mean'], tr.ones(())), plates='plate_Ids')


      Phi = tr['beta']*weather
      #Presence of birds
      tr('z', alan.Bernoulli(logits = Phi))


      tr('alpha', alan.Normal(tr['bird_mean'], tr.ones(())), plates='plate_Ids')
      p = tr['alpha']*quality

      #Observation of birds
      tr('obs', alan.Bernoulli(logits=p*tr['z']), plates='plate_Replicate')






    class Q(alan.AlanModule):
        def __init__(self):
            super().__init__()
            #Year
            self.year_m = nn.Parameter(t.zeros((M,), names=('plate_Years',)))
            self.year_scale = nn.Parameter(t.zeros((M,), names=('plate_Years',)))
            #Bird
            self.bird_m = nn.Parameter(t.zeros((M,J), names=('plate_Years','plate_Birds')))
            self.bird_scale = nn.Parameter(t.zeros((M,J), names=('plate_Years','plate_Birds')))
            #Beta
            self.beta_m = nn.Parameter(t.zeros((M,J,I), names=('plate_Years','plate_Birds', 'plate_Ids')))
            self.beta_scale = nn.Parameter(t.zeros((M,J,I), names=('plate_Years','plate_Birds', 'plate_Ids')))
            #z
            self.z_logits = nn.Parameter(t.zeros((M,J,I), names=('plate_Years','plate_Birds', 'plate_Ids')))


            #alpha
            self.alpha_m = nn.Parameter(t.zeros((M,J,I), names=('plate_Years','plate_Birds', 'plate_Ids')))
            self.alpha_scale = nn.Parameter(t.zeros((M,J,I), names=('plate_Years','plate_Birds', 'plate_Ids')))

        def forward(self, tr, weather, quality):

            tr('year_mean', alan.Normal(self.year_m, self.year_scale.exp()))
            tr('bird_mean', alan.Normal(self.bird_m, self.bird_scale.exp()))

            tr('beta', alan.Normal(self.beta_m, self.beta_scale.exp()))
            tr('z', alan.Bernoulli(logits = self.z_logits))

            tr('alpha', alan.Normal(self.alpha_m, self.alpha_scale.exp()))



    covariates = {'weather': t.load('occupancy/data/weather_train_{}.pt'.format(run)).rename('plate_Years', 'plate_Birds', 'plate_Ids').float(),
        'quality': t.load('occupancy/data/quality_train_{}.pt'.format(run)).rename('plate_Years', 'plate_Birds', 'plate_Ids').float()}
    test_covariates = {'weather': t.load('occupancy/data/weather_test_{}.pt'.format(run)).rename('plate_Years', 'plate_Birds', 'plate_Ids').float(),
        'quality': t.load('occupancy/data/quality_test_{}.pt'.format(run)).rename('plate_Years', 'plate_Birds', 'plate_Ids').float()}
    all_covariates = {'weather': t.cat([covariates['weather'],test_covariates['weather']],-1),
        'quality': t.cat([covariates['quality'],test_covariates['quality']],-1)}




    if use_data:
        data = {'obs':t.load('occupancy/data/birds_train_{}.pt'.format(run)).rename('plate_Year', 'plate_Birds', 'plate_Ids','plate_Replicate')}
        # print(data)
        test_data = {'obs':t.load('occupancy/data/birds_test_{}.pt'.format(run)).rename('plate_Year', 'plate_Birds', 'plate_Ids','plate_Replicate')}
        all_data = {'obs': t.cat([data['obs'],test_data['obs']],-2)}

    else:
        model = alan.Model(P)
        all_data = model.sample_prior(inputs = all_covariates, platesizes= {'plate_Years': M, 'plate_Birds':J, 'plate_Ids':I*2, 'plate_Replicate': 5})
        #data_prior_test = model.sample_prior(platesizes = sizes, inputs = test_covariates)
        data = all_data
        test_data = {}
        data['obs'], test_data['obs'] = t.split(all_data['obs'].clone(), [I,I], -1)

        all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1)}

    return P, Q, data, covariates, all_data, all_covariates, sizes


if __name__ == "__main__":
    seed_torch(0)
    P, Q, data, covariates, all_data, all_covariates, sizes = generate_model(2,2, t.device("cpu"), run=0, use_data=True)


    model = alan.Model(P, Q())
    data = {'obs':data.pop('obs')}
    K = 10
    opt = t.optim.Adam(model.parameters(), lr=0.1)
    for j in range(2000):
        opt.zero_grad()
        sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))
        p_obj, q_obj = sample.rws()
        (p_obj - q_obj).backward()
        opt.step()



        for i in range(10):
            try:
                sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))
                pred_likelihood = model.predictive_ll(sample, N = 10, data_all=all_data, inputs_all=all_covariates)
                break
            except:
                pred_likelihood = 0

        if j % 100 == 0:
            print(f'Elbo: {p_obj.item()}')
            print(f'Pred_ll: {pred_likelihood}')
