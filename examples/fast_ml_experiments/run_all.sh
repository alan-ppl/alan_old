# python runner.py dataset=bus_breakdown model=bus_breakdown training.pred_ll.do_pred_ll=True training.N=2 training.M=2 training.num_iters=500
python runner_ml.py dataset=bus_breakdown model=bus_breakdown training.pred_ll.do_pred_ll=True training.lr=1e-2 training.ML=1 training.num_iters=500
python runner_ml.py dataset=movielens model=movielens training.pred_ll.do_pred_ll=True training.lr=1e-1 training.ML=1 training.N=5 training.M=300
python runner.py dataset=movielens model=movielens training.pred_ll.do_pred_ll=True training.ML=1 training.N=5 training.M=300
