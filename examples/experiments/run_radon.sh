# lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py local=True dataset=radon model=radon_unif_alongreadings training.inference_method=elbo_tmc_new training.num_iters=50000
lbatch -c 1 -g 1 -m 22 -t 25 -a cosc020762 -q cnu --cmd python runner.py dataset=radon model=radon_discrete training.inference_method=rws_tmc training.num_iters=50000
lbatch -c 1 -g 1 -m 22 -t 35 -a cosc020762 -q cnu --cmd python runner.py dataset=radon model=radon_unif_alongreadings training.inference_method=rws_tmc_new training.num_iters=50000
lbatch -c 1 -g 1 -m 22 -t 25 -a cosc020762 -q cnu --cmd python runner.py dataset=radon model=radon_unif_alongreadings training.inference_method=rws_global training.num_iters=50000
