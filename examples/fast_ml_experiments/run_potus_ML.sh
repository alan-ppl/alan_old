lbatch -c 1 -g 1 -m 124 --gputype A100  -t 12 -q cnu -a cosc020762 --cmd python runner_ml.py dataset=potus model=potus training.lr=3e-1 training.ML=1 training.N=5 training.M=300 training.num_iters=750 training.Ks=[5]
lbatch -c 1 -g 1 -m 124 --gputype A100  -t 12 -q cnu -a cosc020762 --cmd python runner_ml.py dataset=potus model=potus training.lr=1e-1 training.ML=1 training.N=5 training.M=300 training.num_iters=750 training.Ks=[5]
lbatch -c 1 -g 1 -m 124 --gputype A100  -t 12 -q cnu -a cosc020762 --cmd python runner_ml.py dataset=potus model=potus training.lr=3e-2 training.ML=1 training.N=5 training.M=300 training.num_iters=750 training.Ks=[5]
lbatch -c 1 -g 1 -m 124 --gputype A100  -t 12 -q cnu -a cosc020762 --cmd python runner_ml.py dataset=potus model=potus training.lr=1e-2 training.ML=1 training.N=5 training.M=300 training.num_iters=750 training.Ks=[5]

# lbatch -c 1 -g 1 -m 124 --gputype A100 --exclude_40G_A100-t 6 -q cnu -a cosc020762 --cmd python runner_ml.py dataset=potus model=potus training.lr=1e-1 training.ML=1 training.N=5 training.M=300 training.num_iters=750 training.Ks=[3,10,30] use_data=False
# lbatch -c 1 -g 1 -m 124 --gputype A100 --exclude_40G_A100-t 6 -q cnu -a cosc020762 --cmd python runner_ml.py dataset=potus model=potus training.lr=1e-2 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 training.Ks=[3,10,30] use_data=False
# lbatch -c 1 -g 1 -m 124 --gputype A100 --exclude_40G_A100-t 6 -q cnu -a cosc020762 --cmd python runner_ml.py dataset=potus model=potus training.lr=3e-2 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 training.Ks=[3,10,30] use_data=False
# lbatch -c 1 -g 1 -m 124 --gputype A100 --exclude_40G_A100-t 6 -q cnu -a cosc020762 --cmd python runner_ml.py dataset=potus model=potus training.lr=1e-3 training.ML=1 training.N=5 training.M=300 training.num_iters=1000 training.Ks=[3,10,30] use_data=False