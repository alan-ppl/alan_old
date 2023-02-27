# ### LIW
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=movielens model=movielens training.inference_method=elbo_tmc_new training.N=5 training.M=150
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=movielens model=movielens training.inference_method=elbo_tmc_new training.N=5 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=movielens model=movielens training.inference_method=elbo_tmc_new training.N=5 training.M=300
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=movielens model=movielens training.inference_method=elbo_tmc_new training.N=10 training.M=150
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=movielens model=movielens training.inference_method=elbo_tmc_new training.N=10 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=movielens model=movielens training.inference_method=elbo_tmc_new training.N=10 training.M=300
#
# ### TMC new
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc_new training.N=5 training.M=150
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc_new training.N=5 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc_new training.N=5 training.M=300
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc_new training.N=10 training.M=150
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc_new training.N=10 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc_new training.N=10 training.M=300
#
# ### Global
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_global training.N=5 training.M=150
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_global training.N=5 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_global training.N=5 training.M=300
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_global training.N=10 training.M=150
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_global training.N=10 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_global training.N=10 training.M=300

# ### RWS
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=rws training.N=5 training.M=150
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=rws training.N=5 training.M=50
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=movielens model=movielens training.inference_method=rws_tmc_new training.N=5 training.M=300
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=rws training.N=10 training.M=150
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=rws training.N=10 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=rws training.N=10 training.M=300
#
# ### RWS Global
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=rws_global training.N=5 training.M=150
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=rws_global training.N=5 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=rws_global training.N=5 training.M=300
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=rws_global training.N=10 training.M=150
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=rws_global training.N=10 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=rws_global training.N=10 training.M=300

# ## tmc
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc training.N=5 training.M=10
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc training.N=5 training.M=150
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc training.N=5 training.M=300
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc training.N=10 training.M=10
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc training.N=10 training.M=150
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=movielens model=movielens training.inference_method=elbo_tmc training.N=10 training.M=300
