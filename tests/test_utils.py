# Let pytest see utils
import sys; sys.path.append(".")
from utils import *
import numpy as np


def named_example_2D() :
    X = t.randn(2, 2)
    X = X.refine_names('_k__a', '_k__b')
    Y = t.randn(2, 2)
    Y = Y.refine_names('_k__a', '_k__b')
    
    return X, Y, '_k__a'


# TODO: this test is broken
def test_logmulmeanexp() :
    X, Y, dim = named_example_2D()
    log_mean_prod_exp = logmulmeanexp(X, Y, dim)
    
    reference = logmmmeanexp(X, Y) \
                .rename(None) \
                .diag()
    stripped = log_mean_prod_exp \
                .rename(None) \
                .squeeze()
    
    #assert(t.allclose(stripped, reference) )



def bivariate_example() :
    x_mean, x_var = -1, 3
    y_mean, y_var = 1, 12
    rho = -0.5
    y = 0
    
    return x_mean, x_var, y_mean, y_var, rho, y 


def test_biv_norm_conditional_mean() :
    x_mean, x_var, y_mean, y_var, rho, y = bivariate_example()
    mu = biv_norm_conditional_mean(x_mean, y_mean, \
                          np.sqrt(x_var), np.sqrt(y_var), \
                          rho, y)
    assert(mu == -0.75)


def test_biv_norm_conditional_var() :
    _, x_var, _, _, rho, _ = bivariate_example()
    var = biv_norm_conditional_var(x_var, rho)
    
    assert(var == 2.25)
    
    
def matrix_example():
    d = 2  
    mean = np.array([[0.], [1.]])
    cov = np.array([
        [1, 0.8], 
        [0.8, 1]
    ])

    mean_x = mean[0,0]
    mean_y = mean[1,0]

    cov00 = cov[0, 0]
    cov11 = cov[1, 1]
    cov01 = cov[0, 1]  
    
    return mean, cov


def matrix_example_2(n=10000) :
    mean = np.array([1, 2, 3, 4])
    cov = np.array(
        [[ 1.0,  0.5,  0.3, -0.1], 
         [ 0.5,  1.0,  0.1, -0.2], 
         [ 0.3,  0.1,  1.0, -0.3], 
         [-0.1, -0.2, -0.3,  0.1]])  # diagonal 
    
    mu1 = mean[0:2].T # Mu of dependent variables
    mu2 = mean[2:4].T # Mu of independent variables
    
    c22 = cov[2:4, 2:4]
    x2 = np.random.multivariate_normal(mu2, c22, n)
    
    return mu1, mu2, cov, x2


def test_conditional_mean() :
    mu1, mu2, cov, x2 = matrix_example_2()
    #c11 = cov[0:2, 0:2] # Covariance matrix of the dependent variables
    c12 = cov[0:2, 2:4] # only containing covariances, not variances
    #c21 = cov[2:4, 0:2] # ""
    c22 = cov[2:4, 2:4] # Covariance matrix of independent variables
    
    mu = conditional_mean(mu1, mu2, c12, c22, x2)
    
    #assert(mu == )


def test_conditional_var() :
    mu1, mu2, cov, x2 = matrix_example_2()
    indices_x1 = (0,2)
    cov = conditional_var(cov, indices_x1)
    
    # assert(cov ==)
    
    