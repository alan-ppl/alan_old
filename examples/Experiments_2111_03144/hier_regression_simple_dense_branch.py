import torch as t
import torch.nn as nn
import alan
from alan.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from alan.backend import vi
import tqdm
from functorch.dim import dims
import numpy as np
import torch.distributions as td
import argparse
import time

print('...', flush=True)

parser = argparse.ArgumentParser(description='Run the Heirarchical regression task.')

parser.add_argument('N', type=int,
                    help='Scale of experiment')
parser.add_argument('K', type=int,
                    help='Number of K samples')

args = parser.parse_args()

t.manual_seed(1)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

theta_size = 10

N = args.N
n_i = 100
plate_1, plate_2 = dims(2 , [N,n_i])
x = t.randn(N,n_i,theta_size)[plate_1,plate_2,:].to(device)

j,k = dims(2)

theta_mean = t.zeros(theta_size).to(device)
theta_sigma = t.ones(theta_size).to(device)

z_sigma = t.ones(theta_size).to(device)

obs_sigma = t.tensor(1).to(device)
def P(tr):
  '''
  Heirarchical Model
  '''
  tr['theta'] = alan.MultivariateNormal(theta_mean, t.diag(theta_sigma))
  tr['z'] = alan.MultivariateNormal(tr['theta'], t.diag(z_sigma), sample_dim=plate_1)

  tr['obs'] = alan.Normal((x @ tr['z']), obs_sigma)


class Q(alan.Q_module):
    def __init__(self):
        super().__init__()
        self.reg_param("theta_mu", t.zeros((theta_size,)))
        self.reg_param("theta_s", t.randn((theta_size,theta_size)))

        self.reg_param("mu", t.zeros((N,theta_size)), [plate_1])
        self.reg_param("A", t.zeros((N,theta_size)), [plate_1])
        self.reg_param("z_s", t.randn((N,theta_size,theta_size)), [plate_1])


    def forward(self, tr):
        sigma_theta = self.theta_s @ self.theta_s.mT
        eye = t.eye(theta_size).to(device)
        sigma_theta = sigma_theta + eye * 0.001

        sigma_z = self.z_s @ self.z_s.mT
        z_eye = eye * 0.001
        sigma_z = sigma_z + z_eye

        tr['theta'] = alan.MultivariateNormal(self.theta_mu, sigma_theta)
        tr['z'] = alan.MultivariateNormal(tr['theta']@self.A + self.mu, sigma_z)







data_y = alan.sample(P,"obs")

## True log prob
##
if N == 10:
    diag = [(t.eye(n_i) + 2 * alan.dename(x).cpu()[i] @ alan.dename(x).cpu()[i].t()) for i in range(N)]

    bmatrix = [[[] for i in range(10)] for n in range (10)]
    for i in range(N):
        for j in range(N):
            if i == j:
                bmatrix[i][i] = diag[i]
            elif j > i:
                bmatrix[i][j] = alan.dename(x).cpu()[i] @ alan.dename(x).cpu()[j].t()
                bmatrix[j][i] = (alan.dename(x).cpu()[i] @ alan.dename(x).cpu()[j].t()).t()




    bmatrix = np.bmat(bmatrix)
    b_matrix = t.from_numpy(bmatrix)
    log_prob = td.MultivariateNormal(t.zeros((b_matrix.shape[0])), b_matrix).log_prob(alan.dename(data_y['obs'].cpu()).flatten())

    print("Log prob: {}".format(log_prob))
    np.save('log_prob.npy', np.array(log_prob))


model = alan.Model(P, Q(), data_y)
model.to(device)

opt = t.optim.Adam(model.parameters(), lr=1E-3)
scheduler = t.optim.lr_scheduler.StepLR(opt, step_size=50000, gamma=0.1)
K=args.K
dim = alan.make_dims(P, K, [plate_1])
print("K={}".format(K))
# start = time.time()
iters = 200000
elbos = []
for i in range(iters):
    opt.zero_grad()
    elbo = model.elbo(dims=dim)
    (-elbo).backward()
    opt.step()
    scheduler.step()
    elbos.append(elbo.item())
    if 0 == i%1000:
        print("Iteration: {0}, ELBO: {1:.2f}".format(i,elbo.item()))



x = x.to('cpu')
#Theta posterior
x_sum = sum([alan.dename(x)[i].t() @ t.inverse(t.eye(n_i) + alan.dename(x)[i] @ alan.dename(x)[i].t()) @ alan.dename(x)[i] for i in range(N)])
y_sum = sum([alan.dename(x)[i].t() @ t.inverse(t.eye(n_i) + alan.dename(x)[i] @ alan.dename(x)[i].t()) @ alan.dename(data_y['obs'].to("cpu"))[i] for i in range(N)])

post_theta_cov = t.eye(theta_size) + x_sum
post_theta_mean = t.inverse(post_theta_cov) @ y_sum

# print('Posterior theta mean')
# print(post_theta_mean)
# print('Approximate Posterior theta mean')
# print(model.Q.theta_mu)
#
# print('Posterior theta cov')
# print(t.round(t.inverse(post_theta_cov),decimals=2))
# print('Approximate Posterior theta cov')
# print(model.Q.log_theta_s.exp())


# print('Posterior z mean')
# print(t.round(post_z_mean, decimals=2).shape)
# print('Approximate Posterior z mean')
# print(t.round(approx_z_mean.reshape(10,-1), decimals=2).shape)
#
# print('Posterior z cov')
# print(t.round(post_z_cov, decimals=2).shape)
# print('Approximate Posterior z cov')
# print(t.round(approx_z_cov, decimals=2).shape)



elbos = np.asarray(elbos)
np.save('Dense_K{0}_N{1}.npy'.format(K, N),elbos)
