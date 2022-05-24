import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm
from torch.distributions import transforms
import torch.distributions as td

device = t.device("cuda" if t.cuda.is_available() else "cpu")

class P(nn.Module):
    def __init__(self, N_obs, N_SN, N_filt, t, fL, dfL, z, t0_mean, J, SNid,
                 Kcor_N, Kcor, fluxscale, duringseason):
        super().__init__()
        self.N_obs = N_obs
        self.N_SN = N_SN
        self.N_filt = N_filt

        self.t = t
        self.fL = fL
        self.dfL = dfL

        self.z = z
        self.t0_mean = t0_mean

        self.J = J
        self.SNid = SNid
        self.Kcor_N = Kcor_N
        self.Kcor = Kcor
        self.fluxscale = fluxscale
        self.duringseason = duringseason

        self.prior_t_hF = t.zero(N_filt, 4)
        self.prior_t_hF_s = t.zero(N_filt, 4)
        self.prior_r_hF = t.zero(N_filt, 4)
        self.prior_r_hF_s = t.zero(N_filt, 4)

    def transform_data(self):

        prior_t_hF[1,0] = -1
        prior_t_hF[1,1] = -0.5
        prior_t_hF[1,2] = 0
        prior_t_hF[1,3] = 0.5
        prior_t_hF[1,4] = 1

        for i in range(N_filt):
            prior_t_hF[0,i] = 0
            prior_t_hF_s[0,i] = 0.1

            prior_t_hF_s[1,i] = 0.1

            prior_t_hF[2,i] = 0
            prior_t_hF_s[2,i] = 0.1

            prior_t_hF[3,i] = 0
            prior_t_hF_s[3,i] = 0.1

            prior_r_hF[0,i] = 0
            prior_r_hF_s[0,i] = 0.1





    def forward(self, tr):




class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_mu = nn.Parameter(t.randn(1))
        self.scale_mu = nn.Parameter(0.1*t.rand(1))

        self.loc_logtau = nn.Parameter(t.zeros(1))
        self.scale_logtau = nn.Parameter(0.1*t.rand(1))


        self.loc_theta=nn.Parameter(t.randn((J,), names=('plate_1',)))
        self.scale_theta = nn.Parameter(0.1*t.rand((J,), names=('plate_1',)))


    def forward(self, tr):
        tr['mu'] = tpp.Normal(self.loc_mu, self.scale_mu.exp())
        tr['tau'] = tpp.LogNormal(self.loc_logtau, self.scale_logtau.exp())

        tr['theta'] = tpp.Normal(self.loc_theta,self.scale_theta.exp())


model = tpp.Model(P, Q(), y)
tpp.sample(P)
model.to(device)
opt = t.optim.Adam(model.parameters(), lr=1E-3)
print("K=10")
for i in range(10):
    opt.zero_grad()
    elbo = model.elbo(K=10)
    (-elbo).backward()
    opt.step()

    if 0 == i%1000:
        print(elbo.item())


print(model.Q.loc_mu)
print(model.Q.scale_mu.exp())
print((model.Q.loc_logtau + 1/2 * model.Q.scale_logtau.exp()).exp())
print(model.Q.loc_theta  + model.Q.loc_mu)
print(model.Q.scale_theta.exp())

print(tpp.sample(Q()))
