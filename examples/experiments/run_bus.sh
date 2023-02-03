lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.inference_method=rws_tmc training.M=3 training.pred_ll.do_pred_ll=True training.num_iters=75000
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.inference_method=rws_tmc_new training.M=3 training.pred_ll.do_pred_ll=True training.num_iters=75000
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.inference_method=rws_global training.M=3 training.pred_ll.do_pred_ll=True training.num_iters=75000
# lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.inference_method=rws_tmc_new local=True training.pred_ll.do_pred_ll=True training.num_iters=50000
