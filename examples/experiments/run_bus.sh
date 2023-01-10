lbatch -c 1 -g 1 -m 22 -t 80 -q cnu --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.inference_method=rws training.pred_ll.do_pred_ll=True
lbatch -c 1 -g 1 -m 22 -t 80 -q cnu --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.inference_method=rws_global training.pred_ll.do_pred_ll=True
