#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export ASCEND_VISIBLE_DEVICES=7
export HF_ENDPOINT=https://hf-mirror.com

FP_MODEL_PATH="/data/oujie/models/llama/Meta-Llama-3.1-8B-Instruct"
QUANT_MODEL_PATH="/data/oujie/models/pre_quantized_models/llama-8b-instruct-w4a4kv4"
OUTPUT_DIR="./log/layer_analysis/llama-8b-instruct-w4a4kv4"

python distill/layer_perturbation_analysis.py \
    --quant_model_path "$QUANT_MODEL_PATH" \
    --fp_model_path "$FP_MODEL_PATH" \
    --model_name llama-8b-instruct-layer-analysis \
    --output_dir "$OUTPUT_DIR" \
    --loss_type mse \
    --hidden_weight 1.0 \
    --attn_weight 0.1 \
    --quant_lr 5e-6 \
    --weight_lr 0 \
    --min_lr_factor 10 \
    --wd 0.01 \
    --train_size 512 \
    --val_size 64 \
    --training_seqlen 1024 \
    --epochs 5 \
    --early_stop 3 \
    --calib_dataset pile \
    --batch_size 4 \
    --seed 2 \
    --ppl_seqlen 2048 \
    --eval_ppl
