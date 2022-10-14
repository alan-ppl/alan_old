import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from movielens_utils import get_ratings, get_features
from functorch.dim import dims
t.manual_seed(0)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# def tril(vec):
#     tril_indices = t.tril_indices(row=10, col=10, offset=0)
#     m = t.zeros((10,10))
#     m[tril_indices[0], tril_indices[1]] = vec
#     return m

x = get_features()
M=10
N=30
plate_1, plate_2 = dims(2 , [M,N])

x = get_features()[:M,:N][plate_1,plate_2]
d_z = 18
def P(tr):
  '''
  Heirarchical Model
  '''

  tr['mu_z'] = tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device))
  tr['psi_z'] = tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device))
  tr['psi_y'] = tpp.Normal(t.zeros(()).to(device), t.ones(()).to(device))

  tr['z'] = tpp.Normal(tr['mu_z'] * t.ones((d_z)).to(device), tr['psi_z'].exp(), sample_dim=plate_1)

  tr['obs'] = tpp.Normal((tr['z'] @ x), tr['psi_y'].exp())



class Q(tpp.Q_module):
    def __init__(self):
        super().__init__()
        #mu_z
        self.reg_param("m_mu_z", t.zeros(()))
        self.reg_param("log_theta_mu_z", t.zeros(()))
        #psi_z
        self.reg_param("m_psi_z", t.zeros(()))
        self.reg_param("log_theta_psi_z", t.zeros(()))
        #psi_y
        self.reg_param("m_psi_y", t.zeros(()))
        self.reg_param("log_theta_psi_y", t.zeros(()))

        #z
        self.reg_param("mu", t.zeros((M,d_z)), [plate_1])
        self.reg_param("log_sigma", t.zeros((M, d_z)), [plate_1])


    def forward(self, tr):
        tr['mu_z'] = tpp.Normal(self.m_mu_z, self.log_theta_mu_z.exp())
        tr['psi_z'] = tpp.Normal(self.m_psi_z, self.log_theta_psi_z.exp())
        tr['psi_y'] = tpp.Normal(self.m_psi_y, self.log_theta_psi_y.exp())

        # sigma_z = self.sigma @ self.sigma.mT
        # eye = t.eye(d_z).to(device)
        # z_eye = eye * 0.001
        # sigma_z = sigma_z + z_eye
        #print(self.mu * t.ones((M,)).to(device)[plate_1])

        tr['z'] = tpp.Normal(self.mu, self.log_sigma.exp())



data_y = tpp.sample(P,"obs")
data_y = {'obs':get_ratings()[:M,:N][plate_1,plate_2]}
K = 5

model = tpp.Model(P, Q(), data_y)
model.to(device)

opt = t.optim.Adam(model.parameters(), lr=1E-3)


dim = tpp.make_dims(P, K, [plate_1])
print("K={}".format(K))
for i in range(50000):
    opt.zero_grad()
    elbo = model.elbo(dims=dim)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print("Iteration: {0}, ELBO: {1:.2f}".format(i,elbo.item()))
