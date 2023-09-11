lbatch -c 1 -g 1 -m 22 -t 6 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens training.lr=1e-1 training.N=5 training.M=300 training.num_iters=10 training.Ks=[10]



