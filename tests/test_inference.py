import sys
sys.path.append('..')

import torch as t
import torch.nn as nn
import tpp
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import unittest

# Testing
class tests(unittest.TestCase):
    def test_plated_gaussian(self):
        '''
        Test posterior inference with a Gaussian with plated observations
        '''
        def P(tr):
          '''
          Bayesian Heirarchical Gaussian Model
          '''
          a = t.zeros(5,)
          tr['mu'] = tpp.MultivariateNormal(a, t.eye(5))
          tr['obs'] = tpp.MultivariateNormal(tr['mu'], t.eye(5), sample_shape=10, sample_names='plate_1')



        class Q(nn.Module):
            def __init__(self):
                super().__init__()
                self.m_mu = nn.Parameter(t.zeros(5,))


                self.log_s_mu = nn.Parameter(t.zeros(5,))


            def forward(self, tr):
                tr['mu'] = tpp.MultivariateNormal(self.m_mu, t.diag(self.log_s_mu.exp()))

        data = {'obs': t.tensor([[ 0.1778, -0.4364, -0.2242, -2.0126, -0.4414],
        [ 2.4132,  0.1589, -0.1721,  1.4035,  0.5189],
        [ 1.9956, -0.6153,  0.3413, -0.9390, -1.6557],
        [ 0.7620,  0.1262, -0.3963,  2.6029,  0.2131],
        [ 1.1981, -0.8900,  0.7388,  0.1689, -1.3313],
        [ 1.7920, -0.4034,  1.1757,  0.4693, -0.5890],
        [ 0.5391,  0.4714,  0.5067,  1.2729,  0.9414],
        [ 1.4357,  0.0208,  0.7751,  1.5554,  0.8555],
        [ 0.1909, -0.3226,  0.5594,  1.0569, -1.6546],
        [-0.1745, -1.9498,  1.5145,  2.7684, -0.8587]],
       names=('plate_1', None))}



        model = tpp.Model(P, Q(), data)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)

        for i in range(10000):
            opt.zero_grad()
            elbo = model.elbo(K=20)
            (-elbo).backward()
            opt.step()

        inferred_mean = model.Q.m_mu

        inferred_cov = model.Q.log_s_mu.exp()

        true_mean = t.mm(t.inverse(t.eye(5) + 1/10 * t.eye(5)),data['obs'].rename(None).mean(axis=0).reshape(-1,1))
        true_cov = t.inverse(t.eye(5) + (1/10 * t.eye(5))) * 1/10


        assert((t.abs(true_mean - inferred_mean.reshape(-1,1))<0.05).all())
        assert((t.abs(true_cov - t.diag(inferred_cov))<0.05).all())

    def test_simple_gaussian(self):
        '''
        Test posterior inference with a Gaussian with a single observation
        '''
        def P(tr):
          '''
          Bayesian Heirarchical Gaussian Model
          '''
          a = t.zeros(5,)
          tr['mu'] = tpp.MultivariateNormal(a, t.eye(5))
          tr['obs'] = tpp.MultivariateNormal(tr['mu'], t.eye(5))



        class Q(nn.Module):
            def __init__(self):
                super().__init__()
                self.m_mu = nn.Parameter(t.zeros(5,))

                self.log_s_mu = nn.Parameter(t.zeros(5,))


            def forward(self, tr):
                tr['mu'] = tpp.MultivariateNormal(self.m_mu, t.diag(self.log_s_mu.exp()))

        data = {'obs': t.tensor([ 0.9004, -3.7564,  0.4881, -1.1412,  0.2087])}

        model = tpp.Model(P, Q(), data)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)

        for i in range(10000):
            opt.zero_grad()
            elbo = model.elbo(K=20)
            (-elbo).backward()
            opt.step()


        inferred_mean = model.Q.m_mu
        inferred_cov = model.Q.log_s_mu.exp()


        true_mean = t.mm(t.inverse(t.eye(5) + t.eye(5)),data['obs'].rename(None).reshape(-1,1))
        true_cov = t.inverse(t.eye(5) + t.eye(5))



        assert((t.abs(true_mean - inferred_mean.reshape(-1,1))<0.05).all())
        assert((t.abs(true_cov - t.diag(inferred_cov))<0.05).all())

    def test_gaussian(self):
        '''
        Test posterior inference with a general gaussian
        '''
        sigma_0 = t.rand(5,5)
        sigma_0 = t.mm(sigma_0, sigma_0.t())
        sigma_0.add_(t.eye(5) * 1e-5)
        sigma = t.rand(5,5)
        sigma = t.mm(sigma, sigma.t())
        sigma.add_(t.eye(5)* 1e-5)
        a = t.randn(5,)
        def P(tr):
          '''
          Bayesian Heirarchical Gaussian Model
          '''

          tr['mu'] = tpp.MultivariateNormal(a, sigma_0)
          tr['obs'] = tpp.MultivariateNormal(tr['mu'], sigma, sample_shape=10, sample_names='plate_1')



        class Q(nn.Module):
            def __init__(self):
                super().__init__()
                self.m_mu = nn.Parameter(t.zeros(5,))

                self.log_s_mu = nn.Parameter(t.randn(5,5))

            def forward(self, tr):
                sigma_nn = t.mm(self.log_s_mu, self.log_s_mu.t())
                sigma_nn.add_(t.eye(5) * 1e-5)
                tr['mu'] = tpp.MultivariateNormal(self.m_mu, covariance_matrix=sigma_nn)

        data = tpp.sample(P, "obs")

        model = tpp.Model(P, Q(), data)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)

        for i in range(30000):
            opt.zero_grad()
            elbo = model.elbo(K=100)
            (-elbo).backward()
            opt.step()


        inferred_mean = model.Q.m_mu
        inferred_cov = t.mm(model.Q.log_s_mu, model.Q.log_s_mu.t())
        inferred_cov.add_(t.eye(5)* 1e-5)


        true_mean = t.mm(sigma_0,t.mm(t.inverse(sigma_0 + 1/10 * sigma),data['obs'].rename(None).mean(axis=0).reshape(-1,1))) + 1/10 * t.mm(sigma,t.mm(t.inverse(sigma_0 + 1/10*sigma),a.reshape(-1,1)))
        true_cov = t.mm(t.mm(sigma_0,t.inverse(sigma_0 + 1/10*sigma)),sigma)


        assert((t.abs(true_mean - inferred_mean.reshape(-1,1))<0.1).all())
        assert(((inferred_cov-true_cov)<0).all())





if __name__ == '__main__':
    unittest.main()
