import sys
sys.path.append('..')
import torch as t
import torch.nn as nn
import tpp
from tpp.backend import vi

from functorch.dim import dims
from tpp.prob_prog import Trace, TraceLogP, TraceSampleLogQ
from tpp.backend import vi
import unittest

# Testing
class tests(unittest.TestCase):
    def test_plated_gaussian(self):
        '''
        Test posterior inference with a Gaussian with plated observations
        '''
        N = 10
        plate_1 = dims(1 , [N])
        def P(tr):
            '''
            Bayesian Gaussian Model
            '''
            a = t.zeros(5,)
            tr['mu'] = tpp.MultivariateNormal(a, t.eye(5))
            tr['obs'] = tpp.Normal(tr['mu'], t.tensor(1), sample_dim=plate_1)



        class Q(nn.Module):
            def __init__(self):
                super().__init__()
                self.m_mu = nn.Parameter(t.zeros(5,))

                self.log_s_mu = nn.Parameter(t.zeros(5,))


            def forward(self, tr):
                tr['mu'] = tpp.MultivariateNormal(self.m_mu, t.diag(self.log_s_mu.exp()))

        data = tpp.sample(P, "obs")

        model = tpp.Model(P, Q(), data)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)

        K = 3
        Ks = tpp.make_dims(P, K)
        for i in range(15000):
            opt.zero_grad()
            elbo = model.elbo(dims=Ks)
            (-elbo).backward()
            opt.step()


        data_obs = tpp.dename(data['obs'])

        b_n = t.mm(t.inverse(t.eye(5) + 1/N * t.eye(5)),data_obs.mean(axis=0).reshape(-1,1))
        A_n = t.inverse(t.eye(5) + 1/N * t.eye(5)) * 1/N

        assert((t.abs(b_n - model.Q.m_mu.reshape(-1,1))<0.1).all())
        assert((t.abs(A_n - t.diag(model.Q.log_s_mu.exp()))<0.1).all())

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
                sigma_nn = self.s_mu @ self.s_mu.mT
                sigma_nn = sigma_nn + t.eye(5) * 1e-5

                tr['mu'] = tpp.MultivariateNormal(self.m_mu, covariance_matrix=sigma_nn)

        data = tpp.sample(P, "obs")

        model = tpp.Model(P, Q(), data)

        opt = t.optim.Adam(model.parameters(), lr=1E-3)

        K=1
        dim = tpp.make_dims(P, K, [plate_1])

        for i in range(15000):
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
                           [0.9,1]]).mT

        data_y = {'obs': t.tensor([ 0.9004, -3.7564,  0.4881, -1.1412,  0.2087,0.478,-1.1])}
        sigma_w = 0.5
        sigma_y = 0.1
        a = t.randn(2,)

        def P(tr):
          '''
          Bayesian Gaussian Linear Model
          '''

          tr['w'] = tpp.MultivariateNormal(a, sigma_w*t.eye(2))
          tr['obs'] = tpp.Normal(tr['w'] @ data_x, sigma_y)



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
