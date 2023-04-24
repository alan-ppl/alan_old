lbatch -c 1 -g 1 -m 124 --exclude_40G_A100 -t 3 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.lr=1e-1 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 plotting.average=True plotting.n_avg=10
lbatch -c 1 -g 1 -m 124 --exclude_40G_A100 -t 3 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.lr=1e-2 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 plotting.average=True plotting.n_avg=10
lbatch -c 1 -g 1 -m 124 --exclude_40G_A100 -t 3 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.lr=3e-2 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 plotting.average=True plotting.n_avg=10
lbatch -c 1 -g 1 -m 124 --exclude_40G_A100 -t 3 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.lr=1e-3 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 plotting.average=True plotting.n_avg=10
lbatch -c 1 -g 1 -m 124 --exclude_40G_A100 -t 3 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.lr=3e-3 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 plotting.average=True plotting.n_avg=10

lbatch -c 1 -g 1 -m 124 --exclude_40G_A100 -t 3 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.lr=1e-1 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 plotting.average=True use_data=False plotting.n_avg=10
lbatch -c 1 -g 1 -m 124 --exclude_40G_A100 -t 3 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.lr=1e-2 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 plotting.average=True use_data=False plotting.n_avg=10
lbatch -c 1 -g 1 -m 124 --exclude_40G_A100 -t 3 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.lr=3e-2 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 plotting.average=True use_data=False plotting.n_avg=10
lbatch -c 1 -g 1 -m 124 --exclude_40G_A100 -t 3 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.lr=1e-3 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 plotting.average=True use_data=False plotting.n_avg=10
lbatch -c 1 -g 1 -m 124 --exclude_40G_A100 -t 3 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.lr=3e-3 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 plotting.average=True use_data=False plotting.n_avg=10
