lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BRWS --cmd python runner_rws.py dataset=bus_breakdown model=bus_breakdown training.lr=1e-2 training.Ks=[1,3,5,10] training.num_iters=2000

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BRWS --cmd python runner_rws.py dataset=bus_breakdown model=bus_breakdown training.lr=1e-1 training.Ks=[1,3,5,10] training.num_iters=2000

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BRWS --cmd python runner_rws.py dataset=bus_breakdown model=bus_breakdown training.lr=5e-2 training.Ks=[1,3,5,10] training.num_iters=2000

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BRWS --cmd python runner_rws.py dataset=bus_breakdown model=bus_breakdown training.lr=5e-1 training.Ks=[1,3,5,10] training.num_iters=2000

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BRWS --cmd python runner_rws.py dataset=bus_breakdown model=bus_breakdown adjust_scale=True training.lr=1e-2 training.Ks=[1,3,5,10] training.num_iters=2000

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BRWS --cmd python runner_rws.py dataset=bus_breakdown model=bus_breakdown adjust_scale=True training.lr=1e-1 training.Ks=[1,3,5,10] training.num_iters=2000

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BRWS --cmd python runner_rws.py dataset=bus_breakdown model=bus_breakdown adjust_scale=True training.lr=5e-2 training.Ks=[1,3,5,10] training.num_iters=2000

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BRWS --cmd python runner_rws.py dataset=bus_breakdown model=bus_breakdown adjust_scale=True training.lr=5e-1 training.Ks=[1,3,5,10] training.num_iters=2000