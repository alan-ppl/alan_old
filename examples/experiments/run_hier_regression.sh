### LIW
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=hier_regression model=hier_regression training.inference_method=elbo training.N=10 training.M=10
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=hier_regression model=hier_regression training.inference_method=elbo training.N=10 training.M=50
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=hier_regression model=hier_regression training.inference_method=elbo training.N=10 training.M=100
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=hier_regression model=hier_regression training.inference_method=elbo training.N=30 training.M=10
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=hier_regression model=hier_regression training.inference_method=elbo training.N=30 training.M=50
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=hier_regression model=hier_regression training.inference_method=elbo training.N=30 training.M=100

### TPP
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo training.N=10 training.M=10
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo training.N=10 training.M=50
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo training.N=10 training.M=100
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo training.N=30 training.M=10
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo training.N=30 training.M=50
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo training.N=30 training.M=100

### Global
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo_global training.N=10 training.M=10
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo_global training.N=10 training.M=50
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo_global training.N=10 training.M=100
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo_global training.N=30 training.M=10
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo_global training.N=30 training.M=50
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo_global training.N=30 training.M=100

# ### TMC
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo_tmc training.N=10 training.M=10
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo_tmc training.N=10 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo_tmc training.N=10 training.M=100
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo_tmc training.N=30 training.M=10
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo_tmc training.N=30 training.M=50
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=hier_regression model=hier_regression training.inference_method=elbo_tmc training.N=30 training.M=100
