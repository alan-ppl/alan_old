### TPP
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo training.N=30 training.M=10
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo training.N=30 training.M=50
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo training.N=30 training.M=100
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo training.N=200 training.M=10
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo training.N=200 training.M=50
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo training.N=200 training.M=100

### Global
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_global training.N=30 training.M=10
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_global training.N=30 training.M=50
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_global training.N=30 training.M=100
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_global training.N=200 training.M=10
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_global training.N=200 training.M=50
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_global training.N=200 training.M=100

### LIW
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=movielens model=movielens training.inference_method=elbo training.N=30 training.M=10
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py llocal=True dataset=movielens model=movielens training.inference_method=elbo training.N=30 training.M=50
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=movielens model=movielens training.inference_method=elbo training.N=30 training.M=100
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=movielens model=movielens training.inference_method=elbo training.N=200 training.M=10
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=movielens model=movielens training.inference_method=elbo training.N=200 training.M=50
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=movielens model=movielens training.inference_method=elbo training.N=200 training.M=100

### tmc
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc training.N=30 training.M=10
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc training.N=30 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc training.N=30 training.M=100
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc training.N=200 training.M=10
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc training.N=200 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc training.N=200 training.M=100
