SEED=927
REPLACE_RATE=0.0

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m MCCWS.script.finetune \
    --batch_size 16 \
    --dropout 0.1 \
    --lr 2e-5 \
    --smooth_factor 0.1 \
    --replace_rate $REPLACE_RATE \
    --epoch 10 \
    --seed $SEED \
    --model_name bert-base-chinese \
    --exp_name testing \
    --trainset_token "[ALL]" \
    --accumulation_step 4 \
    --start_log 280000 \
    --save_step 1000 \
    --gpu 3
