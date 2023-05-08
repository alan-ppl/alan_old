# lbatch -c 1 -g 1 -m 22 -t 3 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens training.lr=1e-1 training.ML=1 training.N=5 training.M=300 training.num_iters=1000
# lbatch -c 1 -g 1 -m 22 -t 3 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens training.lr=1e-2 training.ML=1 training.N=5 training.M=300 training.num_iters=1000
# lbatch -c 1 -g 1 -m 22 -t 3 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens training.lr=3e-2 training.ML=1 training.N=5 training.M=300 training.num_iters=1000
# lbatch -c 1 -g 1 -m 22 -t 3 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens training.lr=1e-3 training.ML=1 training.N=5 training.M=300 training.num_iters=1000

lbatch -c 1 -g 1 -m 22 -t 3 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens training.lr=1e-1 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 use_data=False
lbatch -c 1 -g 1 -m 22 -t 3 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens training.lr=1e-2 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 use_data=False
lbatch -c 1 -g 1 -m 22 -t 3 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens training.lr=3e-2 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 use_data=False
lbatch -c 1 -g 1 -m 22 -t 3 -q cnu -a cosc020762 --cmd python runner.py dataset=movielens model=movielens training.lr=1e-3 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 use_data=False
