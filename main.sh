python main.py \
    --path './testsets/test_Set9/' \
    --bs 1 \
    --sigma 25 \
    --iteration 150000 \
    --lr 1e-4 \
    --model_type 'dropout' \
    --test_frequency 1000 \
    --log_pth './logs/log_dropout.txt' \
    --device 'cuda:0'