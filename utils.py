import torch as t
import numpy as np


def logmeanexp(X, Y):
    xmax = X.max()
    ymax = Y.max()
    X = X - xmax
    Y = Y - ymax
    
    log_exp_prod = (X.exp() * Y.exp()).log()
    
    if len(X.size()) > 1 :
        size = X.size(1)
    else : 
        size = X.size(0)
    
    log_sizes = t.log(t.ones((), device=X.device) * size)
    
    return X + Y + log_exp_prod - log_sizes


# from https://github.com/anonymous-78913/tmc-anon/blob/master/non-fac/model.py
def logmmmeanexp(X, Y):
    xmax = X.max(dim=1, keepdim=True)[0]
    ymax = Y.max(dim=0, keepdim=True)[0]
    X = X - xmax
    Y = Y - ymax
    # NB: need t.matmul instead if broadcasting
    log_exp_prod = t.mm(X.exp(), Y.exp()).log()
    
    return x + y + log_exp_prod \
            - t.log(t.ones((), device=x.device)*X.size(1))



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