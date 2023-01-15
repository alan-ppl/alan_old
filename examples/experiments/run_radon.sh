lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py local=True dataset=radon model=radon_alongreadings training.inference_method=elbo training.num_iters=50000
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongreadings training.inference_method=elbo training.num_iters=50000
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongreadings training.inference_method=elbo_global training.num_iters=50000
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongreadings training.inference_method=rws training.num_iters=50000
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongreadings training.inference_method=rws_global training.num_iters=50000

lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py local=True dataset=radon model=radon_alongzipcodes training.inference_method=elbo training.num_iters=50000
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongzipcodes training.inference_method=elbo training.num_iters=50000
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongzipcodes training.inference_method=elbo_global training.num_iters=50000
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongzipcodes training.inference_method=rws training.num_iters=50000
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongzipcodes training.inference_method=rws_global training.num_iters=50000
