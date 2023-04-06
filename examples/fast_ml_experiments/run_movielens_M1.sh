#ML
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --gputype A100 --cmd python runner_ml.py dataset=movielens model=movielens training.pred_ll.do_pred_ll=True training.lr=1e-1 training.ML=1 training.N=5 training.M=300

#VI
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.pred_ll.do_pred_ll=True training.lr=1e-3 training.num_iters=200 training.N=5 training.M=300
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.pred_ll.do_pred_ll=True training.lr=1e-4 training.num_iters=200 training.N=5 training.M=300
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.pred_ll.do_pred_ll=True training.lr=1e-5 training.num_iters=200 training.N=5 training.M=300
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.pred_ll.do_pred_ll=True training.lr=1e-6 training.num_iters=200 training.N=5 training.M=300
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.pred_ll.do_pred_ll=True training.lr=1e-7 training.num_iters=200 training.N=5 training.M=300
