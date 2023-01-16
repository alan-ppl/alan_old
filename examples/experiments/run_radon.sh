lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py local=True dataset=radon model=radon_alongreadings training.inference_method=elbo training.pred_ll.num_pred_ll_samples=100
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongreadings training.inference_method=elbo training.pred_ll.num_pred_ll_samples=100
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongreadings training.inference_method=elbo_tmc training.pred_ll.num_pred_ll_samples=100
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongreadings training.inference_method=elbo_global training.pred_ll.num_pred_ll_samples=100
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongreadings training.inference_method=rws training.pred_ll.num_pred_ll_samples=100
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongreadings training.inference_method=rws_tmc training.pred_ll.num_pred_ll_samples=100
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongreadings training.inference_method=rws_global training.pred_ll.num_pred_ll_samples=100

lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py local=True dataset=radon model=radon_alongzipcodes training.inference_method=elbo training.pred_ll.num_pred_ll_samples=100
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongzipcodes training.inference_method=elbo training.pred_ll.num_pred_ll_samples=100
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongzipcodes training.inference_method=elbo_tmc training.pred_ll.num_pred_ll_samples=100
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongzipcodes training.inference_method=elbo_global training.pred_ll.num_pred_ll_samples=100
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongzipcodes training.inference_method=rws training.pred_ll.num_pred_ll_samples=100
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongzipcodes training.inference_method=rws_tmc training.pred_ll.num_pred_ll_samples=100
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu --cmd python runner.py dataset=radon model=radon_alongzipcodes training.inference_method=rws_global training.pred_ll.num_pred_ll_samples=100
