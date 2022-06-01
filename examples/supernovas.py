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

        ## Data
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
        self.fluxscale = fluxscale #not used?
        self.duringseason = duringseason

        self.prior_t_hF = t.zero(N_filt, 4)
        self.prior_t_hF_s = t.zero(N_filt, 4)
        self.prior_r_hF = t.zero(N_filt, 4)
        self.prior_r_hF_s = t.zero(N_filt, 4)

        transform_data()

    def transform_data(self):

        self.prior_t_hF[1,0] = -1
        self.prior_t_hF[1,1] = -0.5
        self.prior_t_hF[1,2] = 0
        self.prior_t_hF[1,3] = 0.5
        self.prior_t_hF[1,4] = 1

        self.prior_r_hF[1,0] = 2
        self.prior_r_hF[1,1] = 1
        self.prior_r_hF[1,2] = 0
        self.prior_r_hF[1,3] = -0.5
        self.prior_r_hF[1,4] = -1
        self.prior_r_hF[2,0] = 1
        self.prior_r_hF[2,1] = 0.3
        self.prior_r_hF[2,2] = 0
        self.prior_r_hF[2,3] = -1
        self.prior_r_hF[2,4] = -1

        for i in range(N_filt):
            self.prior_t_hF[0,i] = 0
            self.prior_t_hF_s[0,i] = 0.1

            self.prior_t_hF_s[1,i] = 0.1

            self.prior_t_hF[2,i] = 0
            self.prior_t_hF_s[2,i] = 0.1

            self.prior_t_hF[3,i] = 0
            self.prior_t_hF_s[3,i] = 0.1

            self.prior_r_hF[0,i] = 0
            self.prior_r_hF_s[0,i] = 0.1

            self.prior_r_hF_s[1,i] = 0.1
            self.prior_r_hF_s[2,i] = 0.1

            self.prior_r_hF[3,i] = 0
            self.prior_r_hF_s[3,i] = 0.1

            self.prior_r_hF[4,i] = 0
            self.prior_r_hF_s[4,i] = 0.1

    # def transform_parameters(self, t_hP, sig_t_hP, t_hF, sig_t_hF, t_hSNF, sig_t_hSNF,
    #                         r_hP, sig_r_hP, r_hF, sig_r_hF, r_hSNF, sig_r_hSNF, M_h,
    #                         sig_M_h, M_hF, sig_M_hF, M_hSNF, sig_M_hSNF, Y_h, sig_Y_h,
    #                          Y_hSNF, sig_Y_hSNF, t0s_h, sig_t0s_h, t0s_hSN, sig_t0s_hSN,
    #                          t0l_h, sig_t0l_h, t0l_hSN, sig_t0l_hSN, V_h, V_hF, V_hSNF):
    def transform_parameters(self, tr):
        mm = t.zeros(self.N_obs)
        dm = t.zeros(self.N_obs)
        pt0 = t.zeros(self.N_SN)
        t1 = t.zeros(self.N_SN,self.N_filt)
        t2 = t.zeros(self.N_SN,self.N_filt)
        td = t.zeros(self.N_SN,self.N_filt)
        tp = t.zeros(self.N_SN,self.N_filt)
        lalpha = t.zeros(self.N_SN,self.N_filt)
        lbeta1 = t.zeros(self.N_SN,self.N_filt)
        lbeta2 = t.zeros(self.N_SN,self.N_filt)
        lbetadN = t.zeros(self.N_SN,self.N_filt)
        lbetadC = t.zeros(self.N_SN,self.N_filt)
        Mp = t.zeros(self.N_SN,self.N_filt)
        Md = t.zeros(self.N_SN,self.N_filt)
        V = t.zeros(self.N_SN,self.N_filt)
        M1 = t.zeros(self.N_SN,self.N_filt)
        M2 = t.zeros(self.N_SN,self.N_filt)
        Md = t.zeros(self.N_SN,self.N_filt)

        for l in range(self.N_SN):
            if self.duringseason[l] == 1:
                pt0[l] = -t.exp( tr['t0s_h'] + tr['sig_t0s_h'] * ( tr['t0s_hSN'][l] * tr['sig_t0s_hSN'][l]))
            else:
                pt0[l] = -t.exp( tr['t0l_h'] + tr['sig_t0l_h'] * ( tr['t0l_hSN'][l] * tr['sig_t0l_hSN'][l]))


        for i in range(self.N_filt):
            for j in range(self.N_SN):
                t1[j,i] = t.exp(  log(1) + tr['t_hP'][0] + tr['sig_t_hP'][0] * (tr['t_hF'][0,i] * tr['sig_t_hF'][0,i] \
                                + tr['sig_t_hSNF'][0,(i)*self.N_SN+j] * tr['t_hSNF'][0,(i)*self.N_SN+j]))

                tp[j,i] = t.exp( log(10) + tr['t_hP'][1] + tr['sig_t_hP'][1] * (tr['t_hF'][1,i] * tr['sig_t_hF'][1,i] \
                                + tr['sig_t_hSNF'][1,(i)*self.N_SN+j] * tr['t_hSNF'][1,(i)*self.N_SN+j]))

                t2[j,i] = t.exp( log(100) + tr['t_hP'][2] + tr['sig_t_hP'][2] * (tr['t_hF'][2,i] * tr['sig_t_hF'][2,i] \
                                + tr['sig_t_hSNF'][2,(i)*self.N_SN+j] * tr['t_hSNF'][2,(i)*self.N_SN+j]))

                td[j,i] = t.exp( log(10) + tr['t_hP'][3] + tr['sig_t_hP'][3] * (tr['t_hF'][3,i] * tr['sig_t_hF'][3,i] \
                                + tr['sig_t_hSNF'][3,(i)*self.N_SN+j] * tr['t_hSNF'][3,(i)*self.N_SN+j]))

                lalpha[j,i] = -1 + ( tr['r_hP'][0] + tr['sig_r_hP'][0] * (tr['r_hF'][0,i] * tr['sig_r_hF'][0,i] \
                            + tr['sig_r_hSNF'][0,(i)*self.N_SN+j] * tr['r_hSNF'][0,(i)*self.N_SN+j]))

                lbeta1[j,i] = -4 + ( tr['r_hP'][1] + tr['sig_r_hP'][1] * (tr['r_hF'][1,i] * tr['sig_r_hF'][1,i] \
                            + tr['sig_r_hSNF'][1,(i)*self.N_SN+j] * tr['r_hSNF'][1,(i)*self.N_SN+j]))

                lbeta2[j,i] = -4 + ( tr['r_hP'][2] + tr['sig_r_hP'][2] * (tr['r_hF'][2,i] * tr['sig_r_hF'][2,i] \
                            + tr['sig_r_hSNF'][2,(i)*self.N_SN+j] * tr['r_hSNF'][2,(i)*self.N_SN+j]))

                lbetadN[j,i] = -3 + ( tr['r_hP'][3] + tr['sig_r_hP'][3] * (tr['r_hF'][3,i] * tr['sig_r_hF'][3,i] \
                             + tr['sig_r_hSNF'][3,(i)*self.N_SN+j] * tr['r_hSNF'][3,(i)*self.N_SN+j]))

                lbetadC[j,i] = -5 + ( tr['r_hP'][4] + tr['sig_r_hP'][4] * (tr['r_hF'][4,i] * tr['sig_r_hF'][4,i] \
                             + tr['sig_r_hSNF'][4,(i)*self.N_SN+j] * tr['r_hSNF'][4,(i)*self.N_SN+j]))

                Mp[j,i] = t.exp(tr['M_h'] + tr['sig_M_h'] * (tr['M_hF'][i] * tr['sig_M_hF'][i] \
                        + tr['sig_M_hSNF'][(i)*self.N_SN+j] * tr['M_hSNF'][(i)*self.N_SN+j]))

                Yb[j,i] = tr['Y_h'] + tr['sig_Y_h'] * (tr['Y_hSNF'][(i)*self.N_SN+j] * tr['sig_Y_hSNF'][(i)*self.N_SN+j]);
                V[j,i] = tr['V_h'] * tr['V_hF'][i] * tr['V_hSNF'][(i)*self.N_SN+j]




        M1 = Mp / t.exp( t.exp(lbeta1) * tp )
        M2 = Mp * t.exp( -t.exp(lbeta2) * t2 );
        Md = M2 * t.exp( -t.exp(lbetadN) * td );

        for n in range(self.N_obs):
            j = self.J[n]
            k = self.SNid[n]
            t_exp = ( self.t[n] - (self.t0_mean[k] + pt0[k]) ) / (1 + self.z[k])
            mm_1 = Yb[k,j] if t_exp < 0 else 0

            if ((t_exp>=0) && (t_exp < t1[k,j])):
                mm_2 = Yb[k,j] + M1[k,j] * (t_exp / t1[k,j])**(t.exp(lalpha[k,j]))
            else:
                mm_2 = 0

            if ((t_exp >= t1[k,j]) && (t_exp < t1[k,j] + tp[k,j])):
                mm_3 = Yb[k,j] + M1[k,j] * t.exp(t.exp(lbeta1[k,j]) * (t_exp - t1[k,j]))
            else:
                mm_3 = 0

            if ((t_exp >= t1[k,j] + tp[k,j]) && (t_exp < t1[k,j] + tp[k,j] + t2[k,j])):
                mm_4 = Yb[k,j] + Mp[k,j] * t.exp(-t.exp(lbeta2[k,j]) * (t_exp - t1[k,j] - tp[k,j]))
            else:
                mm_4 = 0

            if ((t_exp >= t1[k,j] + tp[k,j] + t2[k,j]) && (t_exp < t1[k,j] + tp[k,j] + t2[k,j] + td[k,j])):
                mm_5 = Yb[k,j] + M2[k,j] * t.exp(-t.exp(lbetadN[k,j]) * (t_exp - t1[k,j] - tp[k,j] - t2[k,j]))
            else:
                mm_5 = 0

            if ((t_exp >= t1[k,j] + tp[k,j] + t2[k,j] + td[k,j])):
                mm_6 = Yb[k,j] + Md[k,j] * t.exp(-t.exp(lbetadC[k,j]) * (t_exp - t1[k,j] - tp[k,j] - t2[k,j] - td[k,j]))
            else:
                mm_6 = 0

            dm[n] = ((dfL[n]**2) + (V[k,j]**2))**1/2

            if (t_exp<0):
                N_SNc = 0
            elif (t_exp<self.Kcor_N-2):
                Kc_down = 0
                while (Kc_down + 1) < t_exp: # replace this with Kc_down = floor(t_exp - 1)?
                    Kc_down += 1
                Kc_up = Kc_down + 1
                N_SNc = self.Kcor[k,j,Kc_down] + (t_exp - floor(t_exp)) * (self.Kcor[k,j,Kc_up] - self.Kcor[k,j,Kc_down])
            else:
                N_SNc = self.Kcor[k,j,Kcor_N]

            mm[n] = (mm_1 + mm_2 + mm_3 + mm_4 + mm_5 + mm_6) / (10**(N_SNc/(-2.5)))
            return mm, dm

    def forward(self, tr):
        tr['t0s_h'] = tpp.Normal(0, 0.5)
        tr['sig_t0s_h'] = tpp.HalfCauchy(0, 0.1)
        tr['t0l_h'] = tpp.Normal(t.log(100), 1)
        tr['sig_t0l_h'] = tpp.HalfCauchy(0, 0.1)
        tr['V_h'] = tpp.HalfCauchy(0, 0.001)
        tr['Y_h'] = tpp.Normal(0, 0.1)
        tr['sig_Y_h'] = tpp.HalfCauchy(0, 0.01)
        tr['M_h'] = tpp.Normal(0, 1)
        tr['sig_M_h'] = tpp.HalfCauchy(0, 0.1)
        tr['t_hP'] = tpp.Normal(0, 0.1)
        tr['sig_t_hP'] = tpp.HalfCauchy(0, 0.1)
        #plate 1 - 4 samples
        tr['t_hF'] = tpp.MultivariateNormal(self.prior_t_hF, self.prior_t_hF_s)
        tr['sig_t_hF'] = tpp.HalfCauchy(0, 0.1, sample_shape=4)
        tr['t_hSNF'] = tpp.Normal(0,1, sample_shape=4)
        rt['sig_t_hSNF'] = tpp.HalfCauchy(0,0.1, sample_shape=4)

        tr['r_hP'] = tpp.Normal(0,1)
        tr['sig_r_hP'] = tpp.HalfCauchy(0,0.1)

        #plate 2 - 5 samples
        tr['r_hF'] = tpp.MultivariateNormal(self.prior_r_hF, self.prior_r_hF_s)
        tr['sig_r_hF'] = tpp.HalfCauchy(0,0.1, sample_shape=5)
        tr['r_hSNF'] = tpp.Normal(0,1, sample_shape=5)
        tr['sig_r_hSNF'] = tpp.HalfCauchy(0,0.1, sample_shape=5)

        tr['M_hF'] = tpp.Normal(0,1)
        tr['sig_M_hf'] = tpp.HalfCauchy(0,0.1)
        tr['sig_M_hF'] ~ tpp.HalfCauchy(0, 0.1)
        tr['M_hSNF'] ~ tpp.Normal(0,1)
        tr['sig_M_hSNF'] ~ tpp.HalfCauchy(0, 0.1)
        tr['Y_hSNF'] ~ tpp.Normal(0,1)
        tr['sig_Y_hSNF'] ~ tpp.HalfCauchy(0, 0.1)
        tr['V_hF'] ~ tpp.HalfCauchy(0, 0.1)
        tr['V_hSNF'] ~ tpp.HalfCauchy(0, 0.1)
        tr['t0s_hSN'] ~ tpp.Normal(0,1)
        tr['sig_t0s_hSN'] ~ tpp.HalfCauchy(0, 0.1)
        tr['t0l_hSN'] ~ tpp.Normal(0,1)
        tr['sig_t0l_hSN'] ~ tpp.HalfCauchy(0, 0.1)
        mm,dm = transform_parameters(tr)
        tr['fL'] ~ tpp.Normal(mm,dm)




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
