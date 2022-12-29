from alan.ml_updates import *

if __name__ == "__main__":
    N = 10
    MLBernoulli.test(N)
    MLPoisson.test(N)
    MLNormal.test(N)
    MLExponential.test(N)
    MLDirichlet.test(N)
    MLGamma.test(N)
