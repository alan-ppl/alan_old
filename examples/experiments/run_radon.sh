lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py local=True dataset=radon model=radon_unif_alongreadings training.inference_method=elbo training.pred_ll.do_pred_ll=False training.num_iters=50000
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_unif_alongreadings training.inference_method=elbo training.pred_ll.do_pred_ll=False training.num_iters=50000
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_unif_alongreadings training.inference_method=elbo_global training.pred_ll.do_pred_ll=False training.num_iters=50000
