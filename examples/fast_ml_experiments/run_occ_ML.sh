lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n occ_ML2 --cmd python runner_ml.py training.ML=2 dataset=occupancy model=occupancy training.num_iters=4000 training.Ks=[1,3,5,10] training.lr=1e-1

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n occ_ML2 --cmd python runner_ml.py training.ML=2 dataset=occupancy model=occupancy training.num_iters=4000 training.Ks=[1,3,5,10] training.lr=5e-1

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n occ_ML2 --cmd python runner_ml.py training.ML=2 dataset=occupancy model=occupancy training.num_iters=4000 training.Ks=[1,3,5,10] training.lr=1e-2

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n occ_ML2 --cmd python runner_ml.py training.ML=2 dataset=occupancy model=occupancy training.num_iters=4000 training.Ks=[1,3,5,10] training.lr=5e-2

