lbatch -c 1 -g 1 -m 22 -t 6 -q cnu -a cosc020762 --cmd python runner_ml.py dataset=movielens model=movielens training.ML=1 training.N=5 training.M=300 training.num_iters=750 training.Ks=[3,10,30,100]


lbatch -c 1 -g 1 -m 22 -t 6 -q cnu -a cosc020762 --cmd python runner_ml.py dataset=movielens model=movielenstraining.ML=1 training.N=5 training.M=300 training.num_iters=750 training.Ks=[3,10,30,100] use_data=False

