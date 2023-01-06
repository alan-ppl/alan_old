### LIW
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo training.N=2 training.M=2
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo training.N=2 training.M=4
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo training.N=2 training.M=10

### TPP
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo training.N=2 training.M=2
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo training.N=2 training.M=4
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo training.N=2 training.M=10

### Global
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo_global training.N=2 training.M=2
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo_global training.N=2 training.M=4
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo_global training.N=2 training.M=10

# ### TMC
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo_tmc training.N=2 training.M=2
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo_tmc training.N=2 training.M=4
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo_tmc training.N=2 training.M=10
