import torch as t

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