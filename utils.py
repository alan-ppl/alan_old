import torch as t
import numpy as np


def get_number_params(torch_model) :
    params = torch_model.parameters()
    return sum(p.numel() for p in params \
               if p.requires_grad)


def shape_check(d) :
    for k, v in d.items() :
        print(v.shape)


def max_k(T, k) :    
    return T.max(dim=k, keepdim=True)[0]


def denominator(T, i) :
    return t.ones((), device=T.device) * T.size(i)


# Stable general tensor product. Part of the inner product.
def logmulmeanexp(X, Y, dim):
    """
        :param X: tensor of log probabilities
        :param Y: tensor of log probabilities, possibly placeholder
        :param dim: dimension to average over
        
        X and Y have their names aligned beforehand
    """
    assert(X.names == Y.names)
    
    xmax = max_k(X, dim)           
    ymax = max_k(Y, dim)

    X = X - xmax
    Y = Y - ymax
    
    # prod happens in probability space, not log space
    # * is hadamard
    log_mean_prod_exp = (X.exp() * Y.exp()) \
                        .mean(dim, keepdim=True) \
                        .log()

    return log_mean_prod_exp + xmax + ymax 
      


# 2D case
# from https://github.com/anonymous-78913/tmc-anon/blob/master/non-fac/model.py
def logmmmeanexp(X, Y):
    xmax = X.max(dim=1, keepdim=True)[0]
    ymax = Y.max(dim=0, keepdim=True)[0]
    X = X - xmax
    Y = Y - ymax
    # NB: need t.matmul instead if broadcasting
    log_exp_prod = t.mm(X.exp(), Y.exp()).log()
    
    return xmax + ymax + log_exp_prod \
            - t.log(t.ones((), device=xmax.device)*X.size(1))


def logmmmeanexp(X, Y):
    x = X.max(dim=1, keepdim=True)[0]
    y = Y.max(dim=0, keepdim=True)[0]
    X = X - x
    Y = Y - y
    return x + y + t.mm(X.exp(), Y.exp()).log() - t.log(t.ones((), device=x.device)*X.size(1))


# mu_{x1|x2}. just bivariate norm
def biv_norm_conditional_mean(mu1, mu2, sigma1, sigma2, rho, x2) :
    score2 = (x2 - mu2) / sigma2
    return mu1 + sigma1 * rho * score2


def biv_norm_conditional_var(var1, rho) :
    assert(var1 >= 0)
    assert(abs(rho) <= 1)
    
    return var1 - (1 - rho**2)



def generate_linear_data(N, w, sigma, interval) :
    X = np.random.uniform(low=interval[0], \
                          high=interval[1], \
                          size=(N, 1))
    noise = np.random.normal(size=(N, 1), scale=sigma)
    Y = w * X + noise
    
    return X, Y


# Analytical solution for lognormal
# gives log_probs, likelihood
def log_norm(x, mu, std):    
    var = std**2
    norm_constant = -0.5 * t.log(2*np.pi*var)
    sqerror = (x - mu)**2
    prec = 1/var
    
    return norm_constant - (0.5 * prec * sqerror)


# univariate OLS
def analytical_posterior_var(var, X) :
    scaled_prec = (1/var**2) * np.matmul(X.T, X) + 1
    
    return scaled_prec**-1


def analytical_posterior_mean(prior_mean, var, X, Y) :
    scaled_cov = (1/var**2) * np.matmul(X.T, Y)
    post_var = analytical_posterior_var(var, X)
    
    return post_var * (prior_mean + scaled_cov)


# $$\mu_{x|z} = \mathbf{\mu_x + \Sigma_{xz}\Sigma_{zz}^{-1}(z-\mu_z)}$$ 
# x2 is data drawn from the conditioning var
# TODO: consider `solve` over `inv`
def conditional_mean(mu1, mu2, c12, c22, x2) :
    return mu1 + c12.dot(np.linalg.inv(c22)) \
                    .dot((x2 - mu2).T) \
                    .T 

    
# $$\Sigma_{x|z} = \mathbf{\Sigma_{xx} - \Sigma_{xz}\Sigma_{zz}^{-1}\Sigma_{zx}}$$
def conditional_var(cov, dep_indices) :
    i,j = dep_indices
    return np.linalg.inv( np.linalg.inv(cov)[i:j, i:j] )