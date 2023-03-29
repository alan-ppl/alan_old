#ML
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --gputype A100 --cmd python runner_ml.py dataset=bus_breakdown model=bus_breakdown training.pred_ll.do_pred_ll=True training.lr=1e-1 training.ML=1 training.num_iters=1000

#VI
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.pred_ll.do_pred_ll=True training.lr=1e-3
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.pred_ll.do_pred_ll=True training.lr=1e-4
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.pred_ll.do_pred_ll=True training.lr=1e-5
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.pred_ll.do_pred_ll=True training.lr=1e-6
lbatch -c 1 -g 1 -m 22 -t 25 -q cnu -a cosc020762 --gputype A100 --cmd python runner.py dataset=bus_breakdown model=bus_breakdown training.pred_ll.do_pred_ll=True training.lr=1e-7 
