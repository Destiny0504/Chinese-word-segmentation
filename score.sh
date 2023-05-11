START=401000
END=401000
STEP=1000
GPU=0
EXP_NAME=testing

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m MCCWS.inference.infer_all \
        --model_name bert-base-chinese \
        --exp_name $EXP_NAME \
        --dataset_token "[AS]" \
        --testset_token "[AS]" \
        --criteria 10 \
        --start_checkpoint $START \
        --end_checkpoint $END \
        --checkpoint_step $STEP \
        --gpu $GPU
