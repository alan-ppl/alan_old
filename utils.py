import torch as t
import numpy as np



def max_k(T, k) :
    if k is not None :
        m = T.max(dim=k, keepdim=True)[0]
    else :
        m = t.ones(())
    
    return m


def denominator(T, i) :
    return t.ones((), device=T.device) * T.size(i)


# Stable general tensor inner product
# TODO: unfinished
def logmulexp(X, Y, dim):
    """
        :param X: tensor of log probabilities
        :param Y: tensor of log probabilities, possibly placeholder
        :param dim: dimension to average over
        
        X and Y have their names aligned beforehand
    """
    assert(X.names == Y.names)
    
    # 1. get max of each dimension
    # TODO: max of every dimension except dim?
    xmaxs = [ max_k(X, k) \
                for k in X.names]
    xsum = sum(xmaxs)
    ymaxs = [ max_k(Y, k) \
                for k in Y.names]
    ysum = sum(ymaxs)

    X = X - xsum
    Y = Y - ysum
    
    # matmul happens in probability space, not log space
    # hence exp first
    log_exp_prod = (X.exp() * Y.exp()).log()
    
    # then sum out dim? 
    # No: prevents next iteration
    #log_sum = log_exp_prod.sum(dim) \
    #            .log()
    
    # normalise via minus log
    log_size = t.log(denominator(X, dim))
    
    return log_exp_prod + xsum + ysum - log_size
      


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



# mu_{x1|x2}. just bivariate norm
def biv_norm_conditional_mean(mu1, mu2, sigma1, sigma2, rho, x2) :
    score2 = (x2 - mu2) / sigma2
    return mu1 + sigma1 * rho * score2


def biv_norm_conditional_var(var1, rho) :
    assert(var1 >= 0)
    assert(abs(rho) <= 1)
    
    return var1 - (1 - rho**2)


def analytical_posterior_var(var, X) :
    scaled_prec = (1/var**2) * X.T @ X +1
    
    return scaled_prec**-1


def analytical_posterior_mean(prior_mean, var, X, Y) :
    scaled_cov = (1/var**2) * X.T @ Y
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