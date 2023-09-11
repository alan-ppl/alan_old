lbatch -c 1 -g 1 -m 22 -t 6 -q cnu -a cosc020762 --cmd python runner_ml.py dataset=movielens model=movielens training.ML=1 training.N=5 training.M=300 training.num_iters=10 training.Ks=[10] training.decay=0.9 

lbatch -c 1 -g 1 -m 22 -t 6 -q cnu -a cosc020762 --cmd python runner_ml.py dataset=movielens model=movielens training.ML=2 training.N=5 training.M=300 training.num_iters=10 training.Ks=[10] training.decay=0.9 



