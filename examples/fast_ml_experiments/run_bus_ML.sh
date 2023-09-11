lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 --cmd python runner_ml.py training.ML=1 dataset=bus_breakdown model=bus_breakdown training.num_iters=5000 training.Ks=[10]

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 --cmd python runner_ml.py training.ML=2 dataset=bus_breakdown model=bus_breakdown training.num_iters=5000 training.Ks=[10]


