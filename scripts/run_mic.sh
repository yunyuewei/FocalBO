python ../optimization_loop.py --model $1 --algo $2 --acqf ei  --induce_size 50 --max_loop_num 3 --init_num 2000 --eval_num 2500 --batch_size 10 --task Michalewicz
# python optimization_loop.py --model $1 --algo $2 --acqf ei --use_depth 1 --auto 1 --induce_size 50 --max_loop_num 1 --init_num 2000 --eval_num 2500 --batch_size 10 --task Michalewicz