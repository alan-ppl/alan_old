lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=radon model=radon_alongreadings training.inference_method=elbo 
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=radon model=radon_alongreadings training.inference_method=elbo
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=radon model=radon_alongreadings training.inference_method=elbo_global
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=radon model=radon_alongreadings training.inference_method=rws
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=radon model=radon_alongreadings training.inference_method=rws_global

lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=radon model=radon_alongzipcodes training.inference_method=elbo
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=radon model=radon_alongzipcodes training.inference_method=elbo
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=radon model=radon_alongzipcodes training.inference_method=elbo_global
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=radon model=radon_alongzipcodes training.inference_method=rws
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=radon model=radon_alongzipcodes training.inference_method=rws_global
