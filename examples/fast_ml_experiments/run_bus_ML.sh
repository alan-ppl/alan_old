lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BML2 --cmd python runner_ml.py training.ML=2 dataset=bus_breakdown model=bus_breakdown training.num_iters=2000 training.Ks=[1,3,5,10] training.lr=1e-1

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BML2 --cmd python runner_ml.py training.ML=2 dataset=bus_breakdown model=bus_breakdown training.num_iters=2000 training.Ks=[1,3,5,10] training.lr=5e-1

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BML2 --cmd python runner_ml.py training.ML=2 dataset=bus_breakdown model=bus_breakdown training.num_iters=2000 training.Ks=[1,3,5,10] training.lr=1e-2

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BML2 --cmd python runner_ml.py training.ML=2 dataset=bus_breakdown model=bus_breakdown training.num_iters=2000 training.Ks=[1,3,5,10] training.lr=5e-2

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BML2 --cmd python runner_ml.py training.ML=2 dataset=bus_breakdown model=bus_breakdown training.num_iters=2000 training.Ks=[1,3,5,10] training.decay=0.75

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BML2 --cmd python runner_ml.py training.ML=2 dataset=bus_breakdown model=bus_breakdown adjust_scale=True training.num_iters=2000 training.Ks=[1,3,5,10] training.lr=1e-1

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BML2 --cmd python runner_ml.py training.ML=2 dataset=bus_breakdown model=bus_breakdown adjust_scale=True training.num_iters=2000 training.Ks=[1,3,5,10] training.lr=5e-1

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BML2 --cmd python runner_ml.py training.ML=2 dataset=bus_breakdown model=bus_breakdown adjust_scale=True training.num_iters=2000 training.Ks=[1,3,5,10] training.lr=1e-2

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BML2 --cmd python runner_ml.py training.ML=2 dataset=bus_breakdown model=bus_breakdown adjust_scale=True training.num_iters=2000 training.Ks=[1,3,5,10] training.lr=5e-2

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n BML2 --cmd python runner_ml.py training.ML=2 dataset=bus_breakdown model=bus_breakdown adjust_scale=True training.num_iters=2000 training.Ks=[1,3,5,10] training.decay=0.75