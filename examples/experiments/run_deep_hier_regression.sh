### LIW
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo training.N=4 training.M=2 training.pred_ll.do_pred_ll=False
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo training.N=4 training.M=4 training.pred_ll.do_pred_ll=False
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py local=True dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo training.N=4 training.M=10 training.pred_ll.do_pred_ll=False

### TPP
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo training.N=4 training.M=2
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo training.N=4 training.M=4
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo training.N=4 training.M=10

### Global
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo_global training.N=4 training.M=2
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo_global training.N=4 training.M=4
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo_global training.N=4 training.M=10

### RWS
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=rws training.N=4 training.M=2
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=rws training.N=4 training.M=4
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=rws training.N=4 training.M=10

### RWS Global
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=rws_global training.N=4 training.M=2
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=rws_global training.N=4 training.M=4
lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=rws_global training.N=4 training.M=10

# ### TMC
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo_tmc training.N=4 training.M=2
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo_tmc training.N=4 training.M=4
# lbatch -c 1 -g 1 -m 22 -t 10 -q cnu --cmd python runner.py dataset=deep_hier_regression model=deep_hier_regression training.inference_method=elbo_tmc training.N=4 training.M=10