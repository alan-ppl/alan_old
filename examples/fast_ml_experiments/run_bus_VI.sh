# lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=1e-2 training.Ks=[3,10,30] training.num_iters=10000
# lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=8e-3 training.Ks=[3,10,30] training.num_iters=10000
# lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=1e-3 training.Ks=[3,10,30] training.num_iters=10000
# lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=3e-3 training.Ks=[3,10,30] training.num_iters=10000

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=1e-2 training.Ks=[3,10,30] training.num_iters=10000 use_data=False
lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=8e-3 training.Ks=[3,10,30] training.num_iters=10000
lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=1e-3 training.Ks=[3,10,30] training.num_iters=10000 use_data=False
lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=3e-3 training.Ks=[3,10,30] training.num_iters=10000 use_data=False
