import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import reweighted_wake_sleep
import tqdm
from functorch.dim import dims
import matplotlib.pyplot as plt

t.manual_seed(0)

if t.cuda.is_available():
  device = t.device('cuda')
else:
  device = t.device('cpu')

print('Using torch version {}'.format(t.__version__))
print('Using {} device'.format(device))

# Training dataset
train_loader = t.utils.data.DataLoader(
    MNIST(root='/home/thomas/Work/data', train=True, download=True,
          transform=transforms.ToTensor()),
    batch_size=100, shuffle=True, pin_memory=False)
# Test dataset
test_loader = t.utils.data.DataLoader(
    MNIST(root='/home/thomas/Work/data', train=True, download=True,
    transform=transforms.ToTensor()),
    batch_size=100, shuffle=True, pin_memory=False)
print('Dataset loaded')

class linear(nn.Module):
    def __init__(self, d_in, d_out, d_hid=100):
        super().__init__()
        self.w_in = nn.Parameter(t.randn(d_hid, d_in) / 10)
        self.w_mid = nn.Parameter(t.randn(d_hid, d_hid) / 10)
        self.w_out = nn.Parameter(t.randn(d_out, d_hid) / 10)

    def forward(self, x):
        x = F.relu(F.linear(x, self.w_in))
        x = F.relu(F.linear(x, self.w_mid))
        return F.linear(x, self.w_out)

class P(nn.Module):
    def __init__(self, d, D=784):
        super().__init__()
        self.prob = nn.Parameter(t.tensor([1/d]*d))

        self.y_obs = linear(1, D)
        self.z_obs = linear(d, D)

    def forward(self, tr):
        # print('P')
        # print(t.softmax(self.prob,0))
        tr['y'] = tpp.Categorical(t.softmax(self.prob,0))
        tr['z'] = tpp.Normal(t.ones(10,), t.ones(10,))
        obs_logits = self.z_obs(tr['z']) + self.y_obs(tr['y'].unsqueeze(0).type(t.FloatTensor))
        tr['obs'] = tpp.ContinuousBernoulli(logits=obs_logits)



class Q(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.prob = nn.Parameter(t.randn(d,))

        self.m_z = nn.Parameter(t.zeros(d,))
        self.log_s_z = nn.Parameter(t.zeros(d,))



    def forward(self, tr, data):
        print('Q')
        print(self.log_s_z.exp())
        tr['y'] = tpp.Categorical(t.softmax(self.prob,0))
        tr['z'] = tpp.Normal(self.m_z, self.log_s_z.exp())


class vae(nn.Module):
    def __init__(self, P, Q, K=10):
        super().__init__()
        self.P = P
        self.Q = Q
        self.K = K

    def rws(self, data):
        #sample from approximate posterior
        trq = TraceSampleLogQ(dims=dim, reparam=False)
        self.Q(trq, data)
        #compute logP
        trp = TraceLogP(trq.sample, data={'obs': data}, dims=dim)

        self.P(trp)
        return reweighted_wake_sleep(trp.log_prob(), trq.log_prob())





d = 10
D = 28 * 28
P = P(d,D)
model = vae(P, Q(d))
model.to(device)
opt = t.optim.Adam(model.parameters(), lr=1e-3)

K=5

dim = tpp.make_dims(P, K)
print("K={}".format(K))

counter = 0
for i in range(1):
    for x, _ in tqdm.tqdm(train_loader):
        x = x.view(-1, D).to(device).refine_names('plate_batch', ...)
        opt.zero_grad()
        theta_loss, phi_loss = model.rws(x)
        (theta_loss+phi_loss).backward()
        opt.step()
        if (counter % 200 == 0):
            print(phi_loss.item())
        counter += 1

images = tpp.sample(P, 'obs')
plt.imshow(images['obs'].detach().numpy().reshape(28,28), cmap="gray")
plt.show()
