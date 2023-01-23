### RWS discrete Global K
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=5 training.M=150 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=5 training.M=50 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=5 training.M=300 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=10 training.M=150 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=10 training.M=50 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=10 training.M=300 training.pred_ll.do_pred_ll=True

### RWS discrete tmc
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc training.N=5 training.M=150 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc training.N=5 training.M=50 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc training.N=5 training.M=300 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc training.N=10 training.M=150 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc training.N=10 training.M=50 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc training.N=10 training.M=300 training.pred_ll.do_pred_ll=True

### RWS discrete tmc new
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc_new training.N=5 training.M=150 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc_new training.N=5 training.M=50 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc_new training.N=5 training.M=300 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc_new training.N=10 training.M=150 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc_new training.N=10 training.M=50 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc_new training.N=10 training.M=300 training.pred_ll.do_pred_ll=True
