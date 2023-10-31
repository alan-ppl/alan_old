lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n occ_RWS --cmd python runner_rws.py dataset=occupancy model=occupancy training.lr=1e-2 training.Ks=[3,5,10] training.num_iters=4000

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n occ_RWS --cmd python runner_rws.py dataset=occupancy model=occupancy training.lr=1e-1 training.Ks=[3,5,10] training.num_iters=4000

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n occ_RWS --cmd python runner_rws.py dataset=occupancy model=occupancy training.lr=5e-2 training.Ks=[3,5,10] training.num_iters=4000

lbatch -c 1 -g 1 -m 22 -t 24 -q cnu -a cosc020762 -n occ_RWS --cmd python runner_rws.py dataset=occupancy model=occupancy training.lr=5e-1 training.Ks=[3,5,10] training.num_iters=4000