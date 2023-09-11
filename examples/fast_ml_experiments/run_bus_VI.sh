lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=1e-2 training.Ks=[10] training.num_iters=5000
