lbatch -c 1 -g 1 -m 124 -t 6 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=1e-4 training.Ks=[3,10,30] training.num_iters=3500
lbatch -c 1 -g 1 -m 124 -t 6 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=3e-4 training.Ks=[3,10,30] training.num_iters=3500
lbatch -c 1 -g 1 -m 124 -t 6 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=1e-2 training.Ks=[3,10,30] training.num_iters=3500
lbatch -c 1 -g 1 -m 124 -t 6 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=1e-3 training.Ks=[3,10,30] training.num_iters=3500
lbatch -c 1 -g 1 -m 124 -t 6 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=3e-3 training.Ks=[3,10,30] training.num_iters=3500

lbatch -c 1 -g 1 -m 124 -t 6 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=1e-4 training.Ks=[3,10,30] training.num_iters=3500 use_data=False
lbatch -c 1 -g 1 -m 124 -t 6 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=3e-4 training.Ks=[3,10,30] training.num_iters=3500 use_data=False
lbatch -c 1 -g 1 -m 124 -t 6 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=1e-2 training.Ks=[3,10,30] training.num_iters=3500 use_data=False
lbatch -c 1 -g 1 -m 124 -t 6 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=1e-3 training.Ks=[3,10,30] training.num_iters=3500 use_data=False
lbatch -c 1 -g 1 -m 124 -t 6 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.lr=3e-3 training.Ks=[3,10,30] training.num_iters=3500 use_data=False 
