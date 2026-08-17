# PrefixQuant：面向华为昇腾 NPU 的大模型量化与蒸馏

## 项目简介

本仓库基于 [PrefixQuant](https://arxiv.org/abs/2410.05265) 进行扩展，面向华为昇腾 NPU 提供大语言模型的低比特量化、量化后蒸馏、精度评测和性能分析能力。

PrefixQuant 的核心思路是在输入序列前加入少量前缀 Token，将模型中稳定出现的异常激活转移到可控位置，再结合 Hadamard 旋转、静态/动态激活量化和逐块参数重构，降低低比特量化带来的精度损失。本仓库在此基础上增加了 `torch_npu` 设备适配和多种蒸馏训练流程，形成如下实验链路：

```text
全精度模型
   │
   ├─ 异常激活分析与前缀 Token 搜索
   ├─ Hadamard 旋转与量化参数初始化
   ▼
低比特量化模型（W/A/KV 可分别配置）
   │
   ├─ Block-AP 逐层重构
   ├─ 单层 / 多层教师蒸馏
   └─ 量化模型微调
   ▼
蒸馏后的量化模型
   │
   ├─ WikiText-2、C4 困惑度评测
   ├─ lm-eval 下游任务评测
   └─ NPU 显存、利用率与推理性能测试
```

## 主要能力

### 1. 权重、激活和 KV Cache 联合量化

量化入口为 `main.py`，可分别设置：

- 权重量化位宽：`--wbits`
- 激活量化位宽：`--input_bits`
- Key/Value Cache 位宽：`--k_bits`、`--v_bits`
- 静态或动态量化：`--input_mode`、`--kv_mode`
- 对称或非对称量化、分组大小、MSE 初始化和激活裁剪
- W8A8KV8、W4A8KV4、W4A4KV4 等常见组合

模型可采用伪量化完成精度实验，也保留了真实整数量化线性层相关实现。当前仓库更适合作为量化算法研究、精度验证和昇腾迁移实验平台；真实低比特算子的性能收益仍取决于实际部署环境中的算子和推理后端支持。

### 2. PrefixQuant 异常值抑制

通过 `--set_prefixed_tokens` 搜索并设置前缀 Token，使异常激活集中到前缀位置；结合下列旋转选项，可以进一步平滑激活分布：

- `--pre_rotate`：对模型执行预旋转
- `--down_online_had`：在 Down Projection 路径应用在线 Hadamard 变换
- `--qk_online_had`：在 Q/K 路径应用在线 Hadamard 变换

`plot_activation.py` 提供线性层输入/输出、异常 Token 位置、逐层异常数量和三维激活分布等可视化功能。

### 3. Block-AP 量化参数重构

`quantize/block_ap.py` 按 Transformer Block 优化量化参数，可使用校准集对量化步长、零点以及可选的浮点权重进行重构。训练过程支持混合精度、梯度缩放、早停、数据缓存和逐层 CPU/NPU 换入换出，以降低设备内存压力。

### 4. 多种量化后蒸馏策略

`distill/` 目录提供以下训练方式：

| 入口 | 用途 |
| --- | --- |
| `train_multi_layer_distill.py` | 将连续多个 Transformer 层作为一个蒸馏单元，对齐教师与学生的中间表示 |
| `train_block_ap_distill.py` | 将 Block-AP 重构与教师蒸馏结合，可跳过指定层 |
| `train_simple_finetune.py` | 不加载教师模型，仅对量化模型执行校准集微调 |
| `layer_perturbation_analysis.py` | 分析不同层量化扰动对模型输出的影响，为蒸馏层选择提供依据 |
| `test_distilled_model.py` | 加载并评测蒸馏后的量化模型 |

蒸馏使用全精度模型作为教师、预量化模型作为学生，通过隐藏状态 MSE 等损失优化量化参数，并可选择小学习率更新学生模型权重。多层蒸馏能够在显存可控的前提下增强跨层误差修复能力。

### 5. 华为昇腾 NPU 适配

仓库已针对昇腾训练和评测流程加入以下适配：

- 使用 `torch_npu` 检测和选择 `npu` 设备
- 支持 `ASCEND_RT_VISIBLE_DEVICES` 和 `ASCEND_VISIBLE_DEVICES`
- 使用 `torch.npu.amp.autocast` 与 NPU `GradScaler` 进行混合精度训练
- 使用 NPU 显存查询、缓存清理和峰值显存统计接口
- 结合 Hugging Face Accelerate 完成多卡模型切分与调度
- 提供基于 `npu-smi info` 的利用率和 HBM 监控脚本
- 在未检测到 `torch_npu` 时，部分流程保留 CUDA 回退路径

## 目录结构

```text
PrefixQuant/
├── main.py                       # 量化、Block-AP 训练、保存与评测主入口
├── eval.py                       # 已量化模型评测
├── eval_normal.py                # 全精度/普通模型评测
├── plot_activation.py            # 激活值与异常 Token 可视化
├── quantize/
│   ├── quantizer.py              # 基础量化器
│   ├── quant_norm.py             # 量化归一化模块
│   ├── int_linear_fake.py        # 伪量化线性层
│   ├── int_linear_real.py        # 真实整数量化线性层相关实现
│   ├── block_ap.py               # 逐块量化参数重构
│   └── triton_utils/             # 量化算子辅助实现
├── distill/
│   ├── train_multi_layer_distill.py
│   ├── train_block_ap_distill.py
│   ├── train_simple_finetune.py
│   ├── layer_perturbation_analysis.py
│   └── *.sh                      # 训练、分析和评测示例
├── utils/                         # 数据、模型、旋转、量化和训练工具
```

## 环境准备

推荐在已安装昇腾驱动、固件和 CANN Toolkit 的 Linux 服务器上运行。首先加载 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ASCEND_VISIBLE_DEVICES=0,1,2,3
```

创建 Python 环境并安装依赖：

```bash
conda create -n prefixquant python=3.9 -y
conda activate prefixquant

pip install -r requirements.txt
```

还需要根据服务器上的 CANN 和 PyTorch 版本安装相匹配的 `torch_npu`。建议安装后先检查环境：

```bash
python -c "import torch, torch_npu; print(torch.__version__); print(torch.npu.is_available()); print(torch.npu.device_count())"
npu-smi info
```

> 注意：原始依赖中包含 `bitsandbytes`、Triton 和 CUDA 版 `fast-hadamard-transform` 等组件。它们不一定能直接在昇腾环境编译或加速；请以当前实验实际调用的代码路径为准，并按 CANN/torch_npu 版本调整依赖。`torch` 与 `torch_npu` 必须使用官方兼容组合。

## 快速开始

以下命令使用占位路径，运行前请替换模型与输出目录。建议从单卡、小校准集和 W8A8KV8 配置开始验证环境，再逐步尝试 W4A4KV4 和多卡训练。

### 1. 执行 W4A4KV4 量化

```bash
python main.py \
  --model_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --model_name llama-8b-instruct \
  --output_dir ./log/llama-8b-instruct-w4a4kv4 \
  --save_quant_dir ./pre_quantized_models/llama-8b-instruct-w4a4kv4 \
  --wbits 4 \
  --input_bits 4 \
  --input_mode static \
  --k_bits 4 \
  --v_bits 4 \
  --kv_group_size 128 \
  --kv_mode static \
  --mse_init \
  --pre_rotate \
  --down_online_had \
  --qk_online_had \
  --set_prefixed_tokens \
  --train_size 512 \
  --val_size 64 \
  --training_seqlen 1024 \
  --epochs 5 \
  --eval_ppl \
  --eval_tasks arc_easy,arc_challenge,hellaswag,winogrande
```

量化目录除模型和 tokenizer 外，还会保存：

- `prefixed_key_values.pth`：前缀 Token 对应的 KV Cache
- `prefixequant_config.json`：位宽、量化模式和旋转等配置

### 2. 多层蒸馏

```bash
python distill/train_multi_layer_distill.py \
  --quant_model_path ./pre_quantized_models/llama-8b-instruct-w4a4kv4 \
  --fp_model_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --model_name llama-8b-instruct-distill \
  --output_dir ./log/distill/llama-8b-instruct-w4a4kv4 \
  --save_distill_dir ./distilled_models/llama-8b-instruct-w4a4kv4 \
  --num_distill_blocks 2 \
  --loss_type mse \
  --hidden_weight 1.0 \
  --attn_weight 0.1 \
  --quant_lr 5e-6 \
  --weight_lr 0 \
  --train_size 512 \
  --val_size 64 \
  --training_seqlen 1024 \
  --batch_size 4 \
  --epochs 10 \
  --early_stop 3 \
  --calib_dataset pile \
  --eval_ppl \
  --eval_tasks arc_easy,arc_challenge,hellaswag,winogrande
```

教师模型和学生模型需要来自同一基础架构，tokenizer、层数和隐藏维度也应保持一致。蒸馏时会同时加载两份模型；若出现 HBM 不足，可减小 `--batch_size`、`--training_seqlen` 或 `--num_distill_blocks`，并限制可见 NPU 数量和单卡内存上限。

### 3. Block-AP + 蒸馏

仓库已经提供示例脚本：

```bash
bash distill/run_block_ap_distill.sh
```

运行前必须修改脚本中的 `QUANT_MODEL_PATH`、`FP_MODEL_PATH`、`SAVE_DISTILL_DIR` 和 NPU 编号。加上 `--use_distill` 时使用教师监督；去掉该参数可作为不带教师的 Block-AP 对照实验。`--skip_layers "28,29,30,31"` 可排除指定层。

### 4. 评测量化模型

```bash
python eval.py \
  --quant_model ./pre_quantized_models/llama-8b-instruct-w4a4kv4 \
  --eval_ppl \
  --eval_tasks arc_easy,arc_challenge,hellaswag,winogrande
```

`--eval_ppl` 会在 WikiText-2 和 C4 上计算困惑度；`--eval_tasks` 通过 `lm_eval` 执行零样本下游任务评测。


## 数据与模型支持

校准数据入口支持 `pile`、`wikitext2`、`c4`、`ptb` 和 `mix`。仓库中已经包含 WikiText-2 与 C4 的本地数据样例/分片，并会在 `cache/` 中缓存整理后的 DataLoader。

模型工具代码覆盖 LLaMA、Mistral、Qwen2、OPT 和 InternLM2 等结构分支；本仓库现有脚本与日志主要围绕 LLaMA 3/3.1、Qwen2/2.5 及部分 DeepSeek-LLaMA 架构模型。不同架构的模块命名、RoPE 和自定义建模代码可能存在差异，新模型接入前应先进行小规模量化与前向验证。

## 推荐实验流程

1. 运行全精度基线评测，记录 PPL 和下游任务得分。
2. 使用 W8A8KV8、小校准集完成单卡冒烟测试。
3. 开启 Prefix Token 和 Hadamard 旋转，验证量化模型可保存、重载和评测。
4. 尝试 W4A8KV4 或 W4A4KV4，并记录精度和峰值 HBM。
5. 使用 `layer_perturbation_analysis.py` 定位敏感层。
6. 对敏感层执行多层蒸馏或 Block-AP 蒸馏。
7. 对比全精度、仅量化、量化后微调和量化后蒸馏四组结果。

## 当前注意事项

- 示例 Shell 脚本包含特定服务器的模型绝对路径和卡号，不能直接迁移到其他机器。
- `requirements.txt` 尚未固定 CANN 与 `torch_npu` 版本，部署时需要手动选择兼容版本。
- 仓库同时保留 NPU 与 CUDA 分支，但主要修改面向昇腾环境；CUDA 回退并不代表所有路径都经过完整验证。
- 多卡切分依赖 Accelerate 的设备映射。若模型在设备间分配异常，应先核对可见设备编号、逻辑卡号映射和 `--max_memory`。
- 量化和蒸馏结果对校准数据、序列长度、随机种子、学习率及 CANN 算子实现较敏感，建议完整记录实验配置和软件版本。

## 项目来源

本仓库建立在 PrefixQuant 工作之上：

```bibtex
@article{prefixquant,
  title={PrefixQuant: Eliminating Outliers by Prefixed Tokens for Large Language Models Quantization},
  author={Chen, Mengzhao and Liu, Yi and Wang, Jiahao and Bin, Yi and Shao, Wenqi and Luo, Ping},
  journal={arXiv preprint arXiv:2410.05265},
  year={2024}
}
```

原项目许可证见仓库根目录 `LICENSE`。使用、修改和分发本仓库代码时，请同时遵循基础模型、数据集及相关第三方组件的许可证要求。
