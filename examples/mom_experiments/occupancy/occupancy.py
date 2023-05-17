import torch as t
import torch.nn as nn
import alan
import numpy as np

from alan.experiment_utils import seed_torch

def generate_model(N,M,device, dataset_seed=0, QModule=False, use_data=True):

    M = 6
    J = 12
    I = 50
    Returns = 5
    sizes = {'plate_Years': M, 'plate_Birds':J, 'plate_Ids':I, 'plate_Replicate': 5}



    def P(tr):
      '''
      Hierarchical Occupancy Model
      '''
      tr.sample('year_mean', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)), plates='plate_Years')

      tr.sample('bird_mean', alan.Normal(tr['year_mean'], (1/np.sqrt(M)) * t.ones(()).to(device)), plates='plate_Birds')

      tr.sample('beta', alan.Normal(tr['bird_mean'], (1/np.sqrt(M*J)) * t.ones(()).to(device)), plates='plate_Ids')


      Phi = tr['beta']*tr['weather']
      #Presence of birds
      tr.sample('z', alan.Bernoulli(logits = Phi))


      tr.sample('alpha', alan.Normal(tr['bird_mean'], (1/np.sqrt(M*J)) * t.ones(()).to(device)), plates='plate_Ids')
      p = tr['alpha']*tr['quality']

      #Observation of birds
      tr.sample('obs', alan.Bernoulli(logits=p*tr['z']), plates='plate_Replicate')



    if not QModule:
        def Q(tr):
          '''
          Hierarchical Occupancy Model
          '''
          tr.sample('year_mean', alan.Normal(t.zeros(()).to(device), t.ones(()).to(device)), plates='plate_Years')

          tr.sample('bird_mean', alan.Normal(tr['year_mean'], (1/np.sqrt(M)) * t.ones(()).to(device)), plates='plate_Birds')

          tr.sample('beta', alan.Normal(tr['bird_mean'], (1/np.sqrt(M*J)) * t.ones(()).to(device)), plates='plate_Ids')


          Phi = tr['beta']*tr['weather']
          #Presence of birds
          tr.sample('z', alan.Bernoulli(logits = Phi))


          tr.sample('alpha', alan.Normal(tr['bird_mean'], (1/np.sqrt(M*J)) * t.ones(()).to(device)), plates='plate_Ids')
          p = tr['alpha']*tr['quality']

          #Observation of birds
          # tr.sample('obs', alan.Bernoulli(logits=p*tr['z']), plates='plate_Replicate')
    else:
        class Q(alan.QModule):
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

            def forward(self, tr):

                tr.sample('year_mean', alan.Normal(self.year_m, self.year_scale.exp()))
                tr.sample('bird_mean',  alan.Normal(self.bird_m, t.tensor((1/np.sqrt(M))).to(device) * self.bird_scale.exp()))

                tr.sample('beta',  alan.Normal(self.beta_m, t.tensor((1/np.sqrt(M*J))).to(device) * self.beta_scale.exp()))
                tr.sample('z', alan.Bernoulli(logits = self.z_logits))

                tr.sample('alpha',  alan.Normal(self.alpha_m, t.tensor((1/np.sqrt(M*J))).to(device) * self.alpha_scale.exp()))



    covariates = {'weather': t.load('data/weather_train_{}.pt'.format(dataset_seed)).rename('plate_Years', 'plate_Birds', 'plate_Ids').float().to(device),
        'quality': t.load('data/quality_train_{}.pt'.format(dataset_seed)).rename('plate_Years', 'plate_Birds', 'plate_Ids').float().to(device)}
    test_covariates = {'weather': t.load('data/weather_test_{}.pt'.format(dataset_seed)).rename('plate_Years', 'plate_Birds', 'plate_Ids').float().to(device),
        'quality': t.load('data/quality_test_{}.pt'.format(dataset_seed)).rename('plate_Years', 'plate_Birds', 'plate_Ids').float().to(device)}
    all_covariates = {'weather': t.cat([covariates['weather'],test_covariates['weather']],-1),
        'quality': t.cat([covariates['quality'],test_covariates['quality']],-1)}





    data = {'obs':t.load('data/birds_train_{}.pt'.format(dataset_seed)).float().rename('plate_Years', 'plate_Birds', 'plate_Ids','plate_Replicate').to(device)}
    # print(data)
    test_data = {'obs':t.load('data/birds_test_{}.pt'.format(dataset_seed)).float().rename('plate_Years', 'plate_Birds', 'plate_Ids','plate_Replicate').to(device)}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']],-2)}


    return P, Q, data, covariates, all_data, all_covariates, sizes


if __name__ == "__main__":
    seed_torch(0)
    P, Q, data, covariates, all_data, all_covariates, sizes = generate_model(2,2, t.device("cpu"), dataset_seed=0, use_data=True)


    model = alan.Model(P, Q())
    data = {'obs':data.pop('obs')}
    K = 3
    opt = t.optim.Adam(model.parameters(), lr=0.03)
    for j in range(2000):
        opt.zero_grad()
        sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))
        p_obj, q_obj = sample.rws()
        (-q_obj).backward()
        opt.step()



        for i in range(10):
            try:
                sample = model.sample_perm(K, data=data, inputs=covariates, reparam=False, device=t.device('cpu'))
                pred_likelihood = model.predictive_ll(sample, N = 10, data_all=all_data, inputs_all=all_covariates)
                break
            except:
                pred_likelihood = 0

        if j % 10 == 0:
            print(f'Elbo: {p_obj.item()}')
            print(f'Pred_ll: {pred_likelihood}')
