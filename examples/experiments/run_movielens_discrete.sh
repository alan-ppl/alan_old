### RWS discrete
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws training.N=30 training.M=10 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws training.N=30 training.M=50 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws training.N=30 training.M=100 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws training.N=200 training.M=10 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws training.N=200 training.M=50 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws training.N=200 training.M=100 training.pred_ll.do_pred_ll=True

### RWS discrete Global K
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=30 training.M=10 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=30 training.M=50 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=30 training.M=100 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=200 training.M=10 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=200 training.M=50 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=200 training.M=100 training.pred_ll.do_pred_ll=True

# ### RWS discrete tmc
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc training.N=30 training.M=10
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc training.N=30 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc training.N=30 training.M=100
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc training.N=200 training.M=10
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc training.N=200 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc training.N=200 training.M=100
