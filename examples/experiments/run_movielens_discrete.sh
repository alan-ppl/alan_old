### RWS discrete Global K
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=5 training.M=150 training.pred_ll.do_pred_ll=True
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=5 training.M=50 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --exclude_40G_A100 -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.num_iters=50000 training.N=200 training.M=50 training.pred_ll.do_pred_ll=True training.Ks=[3,10,30,100,300,1000,3000,10000,30000]
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=10 training.M=150 training.pred_ll.do_pred_ll=True
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=10 training.M=50 training.pred_ll.do_pred_ll=True
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_global training.N=10 training.M=300 training.pred_ll.do_pred_ll=True

### RWS discrete tmc new
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc_new training.N=5 training.M=150 training.pred_ll.do_pred_ll=True
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc_new training.N=5 training.M=50 training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --exclude_40G_A100 -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc_new training.num_iters=50000 training.N=200 training.M=50 training.pred_ll.do_pred_ll=True training.Ks=[3,10,30,100]
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc_new training.N=10 training.M=150 training.pred_ll.do_pred_ll=True
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc_new training.N=10 training.M=50 training.pred_ll.do_pred_ll=True
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws_tmc_new training.N=10 training.M=300 training.pred_ll.do_pred_ll=True
