from tpp.exp_fam_mixin import *

if __name__ == "__main__":
    N = 10
    BernoulliMixin.test(N)
    PoissonMixin.test(N)
    NormalMixin.test(N)
    ExponentialMixin.test(N)
    DirichletMixin.test(N)
    BetaMixin.test(N)
    GammaMixin.test(N)
    InverseGammaMixin.test(N)
