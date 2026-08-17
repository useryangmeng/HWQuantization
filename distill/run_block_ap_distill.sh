#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export ASCEND_VISIBLE_DEVICES=4,5,6,7
export HF_ENDPOINT=https://hf-mirror.com

QUANT_MODEL_PATH="/data/oujie/models/pre_quantized_models/llama-8b-instruct-w4a4kv4"
FP_MODEL_PATH="/data/oujie/models/llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR="./log/block_ap_distill/llama-8b-instruct-w4a4kv4"
SAVE_DISTILL_DIR="/data/oujie/models/distilled_models/llama-8b-instruct-w4a4kv4-block-ap-distill"

python distill/train_block_ap_distill.py \
    --quant_model_path "$QUANT_MODEL_PATH" \
    --fp_model_path "$FP_MODEL_PATH" \
    --model_name llama-8b-instruct-block-ap-distill \
    --output_dir "$OUTPUT_DIR" \
    --save_distill_dir "$SAVE_DISTILL_DIR" \
    --use_distill \
    --loss_type mse \
    --hidden_weight 1.0 \
    --attn_weight 0.1 \
    --skip_layers "28,29,30,31" \
    --quant_lr 5e-5 \
    --weight_lr 5e-6 \
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
