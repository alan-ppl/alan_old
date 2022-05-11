import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import tqdm

torch.manual_seed(0)

if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')

print('Using torch version {}'.format(torch.__version__))
print('Using {} device'.format(device))

# Training dataset
train_loader = torch.utils.data.DataLoader(
    MNIST(root='/home/thomas/Work/data', train=True, download=True,
          transform=transforms.ToTensor()),
    batch_size=100, shuffle=True, pin_memory=False)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    MNIST(root='/home/thomas/Work/data', train=True, download=True,
    transform=transforms.ToTensor()),
    batch_size=100, shuffle=True, pin_memory=False)
print('Dataset loaded')

class linear(nn.Module):
    def __init__(self, d_in, d_out, d_hid=100):
        super().__init__()
        self.w_in = nn.Parameter(torch.randn(d_hid, d_in) / 10)
        self.w_mid = nn.Parameter(torch.randn(d_hid, d_hid) / 10)
        self.w_out = nn.Parameter(torch.randn(d_out, d_hid) / 10)

    def forward(self, x):
        x = F.relu(F.linear(x, self.w_in))
        x = F.relu(F.linear(x, self.w_mid))
        return F.linear(x, self.w_out)


class P(nn.Module):
    def __init__(self, d, D=784):
        super().__init__()
        self.d = d
        self.z1_z2 = linear(d, 2 * d)
        self.z2_z3 = linear(d, 2 * d)
        self.z3_obs = linear(d, D)

    def forward(self, tr):
        d = self.d
        tr['z3'] = tpp.Normal(torch.zeros(d,).to(device), torch.ones(d,).to(device))
        # z2_loc_scale = self.z1_z2(tr['z1'])
        # tr['z2'] = tpp.Normal(z2_loc_scale[..., :d], z2_loc_scale[..., d:].exp() + 1e-5)
        # z3_loc_scale = self.z2_z3(tr['z1'])
        # tr['z3'] = tpp.Normal(z3_loc_scale[..., :d], z3_loc_scale[..., d:].exp() + 1e-5)
        obs_logits = self.z3_obs(tr['z3'])
        tr['obs'] = tpp.ContinuousBernoulli(logits=obs_logits)


class Q(nn.Module):
    def __init__(self, d, D=784):
        super().__init__()
        self.d = d
        self.obs_z3 = linear(D, 2 * d)
        self.z3_z2 = linear(d, 2 * d)
        self.z2_z1 = linear(d, 2 * d)

    def forward(self, tr, data):
        d = self.d
        z3_loc_scale = self.obs_z3(data)
        tr['z3'] = tpp.Normal(z3_loc_scale[..., :d], z3_loc_scale[..., d:].exp() + 1e-5)
        # z2_loc_scale = self.z3_z2(tr['z3'])
        # tr['z2'] = tpp.Normal(z2_loc_scale[..., :d], z2_loc_scale[..., d:].exp() + 1e-5)
        # z1_loc_scale = self.z2_z1(tr['z3'])
        # tr['z1'] = tpp.Normal(z1_loc_scale[..., :d], z1_loc_scale[..., d:].exp() + 1e-5)


class vae(nn.Module):
    def __init__(self, P, Q, K=10):
        super().__init__()
        self.P = P
        self.Q = Q
        self.K = K

    def elbo(self, data):
        #sample from approximate posterior
        trq = TraceSampleLogQ(K=self.K)
        self.Q(trq, data)
        #compute logP
        trp = TraceLogP(trq.sample, data={'obs': data})
        self.P(trp)
        return vi(trp.log_prob(), trq.log_prob())


d = 16
D = 28 * 28
model = vae(P(d), Q(d))
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-5)
counter = 0

for i in range(10):
    for x, _ in tqdm.tqdm(train_loader):
        x = x.view(-1, D).to(device).refine_names('plate_batch', ...)
        opt.zero_grad()
        elbo = model.elbo(x)
        (-elbo).backward()
        opt.step()
        if (counter % 200 == 0):
            print(elbo.item())
        counter += 1
