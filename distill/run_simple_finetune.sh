#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=6,7
export ASCEND_VISIBLE_DEVICES=6,7
export HF_ENDPOINT=https://hf-mirror.com

QUANT_MODEL_PATH="/data/oujie/models/pre_quantized_models/llama-8b-instruct-w4a4kv4"
OUTPUT_DIR="./log/simple_finetune/llama-8b-instruct-w4a4kv4"
SAVE_DISTILL_DIR="/data/oujie/models/distilled_models/llama-8b-instruct-w4a4kv4-simple-finetune"

python distill/train_simple_finetune.py \
    --quant_model_path "$QUANT_MODEL_PATH" \
    --model_name llama-8b-instruct-simple-finetune \
    --output_dir "$OUTPUT_DIR" \
    --save_distill_dir "$SAVE_DISTILL_DIR" \
    --loss_type mse \
    --quant_lr 5e-5 \
    --weight_lr 0.0 \
    --min_lr_factor 10 \
    --wd 0 \
    --train_size 512 \
    --val_size 64 \
    --training_seqlen 1024 \
    --epochs 5 \
    --early_stop 0 \
    --mse_init \
    --mse_init_size 8 \
    --calib_dataset pile \
    --batch_size 4 \
    --seed 2 \
    --ppl_seqlen 2048 \
    --eval_ppl \
    --eval_tasks arc_easy,arc_challenge,hellaswag,winogrande
