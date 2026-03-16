#!/bin/bash

# 初始化CANN环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 指定推理设备
# ASCEND_RT_VISIBLE_DEVICES=6,7
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_ENDPOINT=https://hf-mirror.com

# 模型快照目录
SNAPSHOT_DIR=$(ls -d /mnt/data1/PrefixQuant/models/deepseek-67b | head -1)

if [ -z "$SNAPSHOT_DIR" ]; then
    echo "未找到模型快照目录"
    exit 1
fi

# 任务1函数：量化训练和评估
task1_quantization() {
    echo "开始任务1: 量化训练和评估"
    local start_time=$(date +%s)
    
    # 使用NPU设备代替CUDA
    python main.py \
        --model_path "$SNAPSHOT_DIR" \
        --model_name llama-deepseek-67B \
        --output_dir ./log/llama-deepseek-67B-w8a8kv8 \
        --wbits 8 \
        --input_bits 8 \
        --input_mode static \
        --v_bits 8 \
        --k_bits 8 \
        --kv_group_size 128 \
        --kv_mode static \
        --mse_init \
        --pre_rotate \
        --down_online_had \
        --qk_online_had \
        --set_prefixed_tokens \
        --eval_ppl \
        --eval_tasks arc_easy,arc_challenge,hellaswag,winogrande \
        --save_quant_dir /mnt/data1/PrefixQuant/pre_quantized_models/llama-deepseek-67B-w8a8kv8

    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        echo "量化评估完成!"
        echo "总耗时: $duration 秒"
    else
        echo "量化评估失败，退出码: $exit_code"
        echo "运行时间: $duration 秒"
    fi
    
    return $exit_code
}

# 任务2函数：评估量化模型
task2_evaluation() {
    echo "开始任务2: 评估量化模型"
    local start_time=$(date +%s)
    
    # 使用NPU设备代替CUDA
    python eval.py \
        --quant_model /mnt/data1/PrefixQuant/pre_quantized_models/llama-deepseek-67B-w8a8kv8 \
        --eval_ppl \
        --eval_tasks arc_easy,arc_challenge,hellaswag,winogrande

    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        echo "模型评估完成!"
        echo "总耗时: $duration 秒"
    else
        echo "模型评估失败，退出码: $exit_code"
        echo "运行时间: $duration 秒"
    fi
    
    return $exit_code
}

# 任务3函数：绘制激活分布
task3_plotting() {
    echo "开始任务3: 绘制激活分布"
    local start_time=$(date +%s)
    
    # 使用NPU设备代替CUDA
    python plot_activation.py \
        --model_path "$SNAPSHOT_DIR" \
        --model_name Qwen-1.5B \
        --plot_linear_input

    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        echo "激活分布绘制完成!"
        echo "总耗时: $duration 秒"
    else
        echo "激活分布绘制失败，退出码: $exit_code"
        echo "运行时间: $duration 秒"
    fi
    
    return $exit_code
}

# 执行任务
task1_quantization
# task2_evaluation
task_exit=$?

exit $task_exit