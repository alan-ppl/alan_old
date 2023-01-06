lbatch -c 1 -g 1 -m 22 -t 60 -q cnu --cmd python runner.py local=True dataset=radon model=radon training.inference_method=elbo training.N=2 training.M=2
lbatch -c 1 -g 1 -m 22 -t 60 -q cnu --cmd python runner.py dataset=radon model=radon training.inference_method=elbo training.N=2 training.M=2
lbatch -c 1 -g 1 -m 22 -t 60 -q cnu --cmd python runner.py dataset=radon model=radon training.inference_method=elbo_global training.N=2 training.M=2
# lbatch -c 1 -g 1 -m 22 -t 60 -q cnu --cmd python runner.py dataset=radon model=radon training.inference_method=elbo_tmc training.N=2 training.M=2
