from tpp.exp_fam_mixin import *

if __name__ == "__main__":
    N = 10
    BernoulliProbsMixin.test(N)
    BernoulliLogitsMixin.test(N)
    PoissonMixin.test(N)
    NormalMixin.test(N)
    ExponentialMixin.test(N)
    DirichletMixin.test(N)
    BetaMixin.test(N)
    GammaMixin.test(N)
    MvNormalMixin.test(N)
