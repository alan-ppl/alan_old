import sys
sys.path.append('..')
import torch as t
import torch.nn as nn
import tpp
from tpp.backend import vi

from torchdim import dims
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import unittest

# Testing
class tests(unittest.TestCase):
    def test_plated_gaussian(self):
        '''
        Test posterior inference with a Gaussian with plated observations
        '''
        plate_1 = dims(1 , [10])
        def P(tr):
          '''
          Bayesian Heirarchical Gaussian Model
          '''
          a = t.zeros(5,)
          tr['mu'] = tpp.MultivariateNormal(a, t.eye(5))
          tr['obs'] = tpp.MultivariateNormal(tr['mu'], t.eye(5), sample_dim=plate_1)



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
        [-0.1745, -1.9498,  1.5145,  2.7684, -0.8587]])}



        model = tpp.Model(P, Q(), data)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)

        K = 5
        dim = tpp.make_dims(P, K, [plate_1])

        for i in range(15000):
            opt.zero_grad()
            elbo = model.elbo(dims=dim)
            (-elbo).backward()
            opt.step()

        inferred_mean = model.Q.m_mu

        inferred_cov = model.Q.log_s_mu.exp()

        true_mean = t.mm(t.inverse(t.eye(5) + 1/10 * t.eye(5)),data['obs'].mean(axis=0).reshape(-1,1))
        true_cov = t.inverse(t.eye(5) + (1/10 * t.eye(5))) * 1/10

        assert((t.abs(true_mean - inferred_mean.reshape(-1,1))<0.1).all())
        assert((t.abs(true_cov - t.diag(inferred_cov))<0.1).all())

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
        K = 5
        dim = tpp.make_dims(P, K)

        for i in range(15000):
            opt.zero_grad()
            elbo = model.elbo(dims=dim)
            (-elbo).backward()
            opt.step()


        inferred_mean = model.Q.m_mu
        inferred_cov = model.Q.log_s_mu.exp()


        true_mean = t.mm(t.inverse(t.eye(5) + t.eye(5)),data['obs'].rename(None).reshape(-1,1))
        true_cov = t.inverse(t.eye(5) + t.eye(5))



        assert((t.abs(true_mean - inferred_mean.reshape(-1,1))<0.1).all())
        assert((t.abs(true_cov - t.diag(inferred_cov))<0.1).all())

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

        N = 10
        plate_1 = dims(1 , [N])
        def P(tr):
          '''
          Bayesian Gaussian Model
          '''

          tr['mu'] = tpp.MultivariateNormal(a, sigma_0)
          tr['obs'] = tpp.MultivariateNormal(tr['mu'], sigma, sample_dim=plate_1)



        class Q(nn.Module):
            def __init__(self):
                super().__init__()
                self.m_mu = nn.Parameter(t.zeros(5,))

                self.s_mu = nn.Parameter(t.randn(5,5))

            def forward(self, tr):
                sigma_nn = t.mm(self.s_mu, self.s_mu.t())
                sigma_nn.add_(t.eye(5) * 1e-5)

                tr['mu'] = tpp.MultivariateNormal(self.m_mu, covariance_matrix=sigma_nn)

        data = tpp.sample(P, "obs")

        model = tpp.Model(P, Q(), data)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)

        K=2
        dim = tpp.make_dims(P, K, [plate_1])

        for i in range(20000):
            opt.zero_grad()
            elbo = model.elbo(dims=dim)
            (-elbo).backward()
            opt.step()


        inferred_mean = model.Q.m_mu
        inferred_cov = t.mm(model.Q.s_mu, model.Q.s_mu.t())
        inferred_cov.add_(t.eye(5)* 1e-5)


        y_hat = tpp.dename(data['obs']).mean(axis=0).reshape(-1,1)
        true_cov = t.inverse(N * t.inverse(sigma) + t.inverse(sigma_0))
        true_mean = (true_cov @ (N*t.inverse(sigma) @ y_hat + t.inverse(sigma_0)@a.reshape(-1,1))).reshape(1,-1)


        print(true_cov)

        print(inferred_cov)

        print(true_mean)

        print(inferred_mean)

        assert(((t.abs(true_mean - inferred_mean))<0.5).all())
        assert(((inferred_cov-true_cov)<0.5).all())

    def test_linear_gaussian(self):
        '''
        Testing Linear gaussian
        '''
        data_x = t.tensor([[0.4,1],
                           [0.5,1],
                           [0.24,1],
                           [-0.68,1],
                           [-0.4,1],
                           [-0.3,1],
                           [0.9,1]]).t()

        data_y = {'obs': t.tensor([ 0.9004, -3.7564,  0.4881, -1.1412,  0.2087,0.478,-1.1])}
        sigma_w = 0.5
        sigma_y = 0.1
        a = t.randn(2,)

        def P(tr):
          '''
          Bayesian Gaussian Linear Model
          '''

          tr['w'] = tpp.MultivariateNormal(a, sigma_w*t.eye(2))
          tr['obs'] = tpp.Normal(t.mm(tr['w'], data_x), sigma_y)



        class Q(nn.Module):
            def __init__(self):
                super().__init__()
                self.m_mu = nn.Parameter(t.zeros(2,))


                self.log_s_mu = nn.Parameter(t.randn(2,2))



            def forward(self, tr):
                sigma_nn = t.mm(self.log_s_mu, self.log_s_mu.t())
                sigma_nn.add_(t.eye(2) * 1e-5)
                tr['w'] = tpp.MultivariateNormal(self.m_mu, sigma_nn)




        model = tpp.Model(P, Q(), data_y)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)

        K = 3
        dim = tpp.make_dims(P, K)
        for i in range(15000):
            opt.zero_grad()
            elbo = model.elbo(dims=dim)
            (-elbo).backward()
            opt.step()


        inferred_cov = t.mm(model.Q.log_s_mu, model.Q.log_s_mu.t())
        inferred_cov.add_(t.eye(2)* 1e-5)
        inferred_mean = model.Q.m_mu

        V_n = sigma_y * t.inverse(sigma_y * t.inverse(sigma_w*t.eye(2)) + t.mm(data_x,data_x.t()))
        w_n = t.mm(V_n, t.mm(t.inverse(sigma_w*t.eye(2)),a.reshape(-1,1))) + (1/sigma_y) * t.mm(V_n, t.mm(data_x, data_y['obs'].reshape(-1,1)))

        assert((t.abs(w_n - inferred_mean.reshape(-1,1))<0.2).all())
        assert(((inferred_cov-V_n)<0.2).all())







if __name__ == '__main__':
    unittest.main()
