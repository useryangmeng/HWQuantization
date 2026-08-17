import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.hooks import remove_hook_from_module
from utils.data_utils import get_loaders, test_ppl, BlockTrainDataset, copy_block_dataset
from utils.quant_utils import wrap_to_quant_model, init_weight_quantizer, init_input_quantizer, register_online_had, init_k_quantizer, init_v_quantizer, quant_parameters, weight_parameters, trainable_parameters, set_quant_state, set_quant_parameters, set_weight_parameters, trainable_parameters_num, quant_inplace
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
from utils.train_utils import load_json_as_namespace, create_logger, NativeScalerWithGradNormCount
from accelerate import init_empty_weights, load_checkpoint_in_model
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import gc
from contextlib import nullcontext
from main import evaluate

from distill.layer_distill import multi_layer_distillation_loss

try:
    import torch_npu
    torch.backends.cudnn.benchmark = False
except ImportError:
    print("Warning: torch_npu not found, falling back to CUDA")
    torch.backends.cudnn.benchmark = True


class CustomLRSchedule(object):
    def __init__(self, args, total_iter) -> None:
        param_group_index = 0
        if args.quant_lr > 0:
            empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
            self.quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_iter, eta_min=args.quant_lr/args.min_lr_factor)
            self.quant_index = param_group_index
            param_group_index += 1
        else:
            self.quant_scheduler = None
        if args.weight_lr > 0:
            empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
            self.weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_iter, eta_min=args.weight_lr/args.min_lr_factor)
            self.weight_index = param_group_index
            param_group_index += 1  
        else:
            self.weight_scheduler = None
    
    def step(self, optimizer):
        if self.quant_scheduler is not None:
            self.quant_scheduler.step()
            optimizer.param_groups[self.quant_index]['lr'] = self.quant_scheduler.get_lr()[0]
        if self.weight_scheduler is not None:
            self.weight_scheduler.step()
            optimizer.param_groups[self.weight_index]['lr'] = self.weight_scheduler.get_lr()[0]


@torch.no_grad()
def update_dataset(layers, source_dataset, target_dataset, dev, attention_mask, position_ids, prefixed_key_values):
    try:
        with torch.npu.amp.autocast():
            for index, inps in enumerate(source_dataset):
                inps = inps.to(dev)
                if len(inps.shape) == 2:
                    inps = inps.unsqueeze(0)
                hidden = inps
                for layer in layers:
                    past_key_value = None
                    if prefixed_key_values is not None:
                        try:
                            past_key_value = model_utils.get_kv_cache(prefixed_key_values, bs=source_dataset.batch_size)
                        except:
                            past_key_value = None
                    hidden = layer(hidden, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value)[0]
                target_dataset.update_data(index, hidden.to('cpu'))
    except AttributeError:
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(source_dataset):
                inps = inps.to(dev)
                if len(inps.shape) == 2:
                    inps = inps.unsqueeze(0)
                hidden = inps
                for layer in layers:
                    past_key_value = None
                    if prefixed_key_values is not None:
                        try:
                            past_key_value = model_utils.get_kv_cache(prefixed_key_values, bs=source_dataset.batch_size)
                        except:
                            past_key_value = None
                    hidden = layer(hidden, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value)[0]
                target_dataset.update_data(index, hidden.to('cpu'))


def distill_single_layer(student_model, teacher_model, prefixed_key_values, layer_idx, args, trainloader, valloader, logger):
    logger.info(f"=== Analyzing layer {layer_idx} ===")
    
    try:
        dev = torch.device("npu" if torch.npu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    except AttributeError:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    prefixed_key_values = model_utils.mv_kv_cache(prefixed_key_values, dev=dev)
    use_cache = student_model.config.use_cache
    student_model.config.use_cache = True
    teacher_model.config.use_cache = True
    
    layers_student = student_model.model.layers
    layers_teacher = teacher_model.model.layers
    
    student_model.model.embed_tokens = student_model.model.embed_tokens.to(dev)
    student_model.model.norm = student_model.model.norm.to(dev)
    teacher_model.model.embed_tokens = teacher_model.model.embed_tokens.to(dev)
    teacher_model.model.norm = teacher_model.model.norm.to(dev)
    
    if hasattr(student_model.model, 'rotary_emb'):
        student_model.model.rotary_emb = student_model.model.rotary_emb.to(dev)
    if hasattr(teacher_model.model, 'rotary_emb'):
        teacher_model.model.rotary_emb = teacher_model.model.rotary_emb.to(dev)
    
    dtype = torch.float16 if not args.use_fp32 else torch.float32
    
    try:
        traincast = torch.npu.amp.autocast if not args.use_fp32 else nullcontext
    except AttributeError:
        traincast = torch.cuda.amp.autocast if not args.use_fp32 else nullcontext
    
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None
        
        def forward(self, inp, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None:
                self.position_ids = kwargs["position_ids"]
            raise ValueError
    
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                student_model.config.hidden_size, args.batch_size, dtype, cache_path=args.cache_dir)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                student_model.config.hidden_size, args.batch_size, dtype, cache_path=args.cache_dir)
    
    layers_student[0] = Catcher(layers_student[0], fp_train_inps)
    iters = len(trainloader) // args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size, (i+1)*args.batch_size)], dim=0)
            try:
                student_model(data.to(dev), past_key_values=model_utils.get_kv_cache(prefixed_key_values, bs=args.batch_size))
            except ValueError:
                pass
    attention_mask = layers_student[0].attention_mask
    position_ids = layers_student[0].position_ids
    attention_mask = attention_mask.to(dtype) if attention_mask is not None else None
    layers_student[0] = layers_student[0].module
    
    layers_student[0] = Catcher(layers_student[0], fp_val_inps)
    iters = len(valloader) // args.batch_size if valloader else 0
    if iters > 0:
        with torch.no_grad():
            for i in range(iters):
                data = torch.cat([valloader[j][0] for j in range(i*args.batch_size, (i+1)*args.batch_size)], dim=0)
                try:
                    student_model(data.to(dev), past_key_values=model_utils.get_kv_cache(prefixed_key_values, bs=args.batch_size))
                except ValueError:
                    pass
    layers_student[0] = layers_student[0].module
    
    layers_student[0] = layers_student[0].cpu()
    student_model.model.embed_tokens = student_model.model.embed_tokens.cpu()
    student_model.model.norm = student_model.model.norm.cpu()
    teacher_model.model.embed_tokens = teacher_model.model.embed_tokens.cpu()
    teacher_model.model.norm = teacher_model.model.norm.cpu()
    
    if hasattr(student_model.model, 'rotary_emb'):
        student_model.model.rotary_emb = student_model.model.rotary_emb.cpu()
    if hasattr(teacher_model.model, 'rotary_emb'):
        teacher_model.model.rotary_emb = teacher_model.model.rotary_emb.cpu()
    
    try:
        torch.npu.empty_cache()
    except AttributeError:
        try:
            torch.cuda.empty_cache()
        except AttributeError:
            pass
    
    quant_train_inps = copy_block_dataset(fp_train_inps)
    quant_val_inps = copy_block_dataset(fp_val_inps)
    
    set_quant_state(student_model, weight_quant=True, act_quant=True)
    
    start_layer = layer_idx
    end_layer = layer_idx + 1
    
    fp_block = layers_teacher[start_layer].to(dev)
    quant_block = layers_student[start_layer].to(dev)
    
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    set_quant_state(fp_block, weight_quant=False, act_quant=False)
    
    update_dataset([fp_block], quant_train_inps, fp_train_inps, dev, attention_mask, position_ids, prefixed_key_values)
    if quant_val_inps:
        update_dataset([fp_block], quant_val_inps, fp_val_inps, dev, attention_mask, position_ids, prefixed_key_values)
    
    set_quant_state(quant_block, weight_quant=True, act_quant=True)
    
    with torch.no_grad():
        quant_block.float()
    
    set_quant_parameters(quant_block, args.quant_lr > 0)
    set_weight_parameters(quant_block, args.weight_lr > 0)
    
    param = []
    if args.quant_lr > 0:
        all_quant_params = list(quant_parameters(quant_block))
        param.append({"params": all_quant_params, "lr": args.quant_lr})
    if args.weight_lr > 0:
        all_weight_params = list(weight_parameters(quant_block))
        param.append({"params": all_weight_params, "lr": args.weight_lr})
    
    block_training_iteration = args.epochs * len(quant_train_inps)
    lr_schedule = CustomLRSchedule(args, block_training_iteration)
    optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
    loss_scaler = NativeScalerWithGradNormCount()
    
    trainable_number = trainable_parameters_num(quant_block)
    logger.info(f"Layer {layer_idx} trainable parameter number: {trainable_number/1e6}M")
    
    best_val_loss = 1e6
    early_stop_flag = 0
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss_list = []
        norm_list = []
        
        for idx in range(len(quant_train_inps)):
            with traincast():
                input_data = quant_train_inps[idx].to(dev)
                if len(input_data.shape) == 2:
                    input_data = input_data.unsqueeze(0)
                
                loss = multi_layer_distillation_loss(
                    [fp_block], [quant_block], input_data,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    loss_type=args.loss_type,
                    hidden_weight=args.hidden_weight,
                    attn_weight=args.attn_weight,
                    return_hidden_states=False,
                    return_attentions=False
                )
            
            if not math.isfinite(loss.item()):
                logger.warning("Loss is NAN, skipping this batch")
                continue
            
            train_loss_list.append(loss.detach().cpu())
            optimizer.zero_grad()
            all_trainable_params = list(trainable_parameters(quant_block))
            norm = loss_scaler(loss, optimizer, parameters=iter(all_trainable_params)).cpu()
            norm_list.append(norm.data)
            lr_schedule.step(optimizer)
        
        train_loss = torch.stack(train_loss_list).mean() if train_loss_list else torch.tensor(0.0)
        norm_mean = torch.stack(norm_list).mean() if norm_list else torch.tensor(0.0)
        
        val_loss = torch.tensor(0.0)
        if quant_val_inps:
            val_loss_list = []
            with torch.no_grad():
                for idx in range(len(quant_val_inps)):
                    with traincast():
                        input_data = quant_val_inps[idx].to(dev)
                        if len(input_data.shape) == 2:
                            input_data = input_data.unsqueeze(0)
                        
                        loss = multi_layer_distillation_loss(
                            [fp_block], [quant_block], input_data,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=None,
                            loss_type=args.loss_type,
                            hidden_weight=args.hidden_weight,
                            attn_weight=args.attn_weight,
                            return_hidden_states=False,
                            return_attentions=False
                        )
                    val_loss_list.append(loss.detach().cpu())
            val_loss = torch.stack(val_loss_list).mean() if val_loss_list else torch.tensor(0.0)
        
        train_loss_history.append(train_loss.item())
        val_loss_history.append(val_loss.item())
        
        try:
            max_mem = torch.npu.max_memory_allocated(dev) / 1024**2
        except AttributeError:
            try:
                max_mem = torch.cuda.max_memory_allocated(dev) / 1024**2
            except AttributeError:
                max_mem = 0
        
        logger.info(f"Layer {layer_idx} epoch {epoch} train_loss:{train_loss:.4f} val_loss:{val_loss:.4f} norm:{norm_mean:.8f} max_mem:{max_mem:.2f}MB time:{time.time()-start_time:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_flag = 0
        else:
            early_stop_flag += 1
            if args.early_stop > 0 and early_stop_flag >= args.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        optimizer.zero_grad()
    
    del optimizer, param
    
    quant_block.half()
    set_quant_state(quant_block, weight_quant=False, act_quant=True)
    
    update_dataset([quant_block], quant_train_inps, quant_train_inps, dev, attention_mask, position_ids, prefixed_key_values)
    if quant_val_inps:
        update_dataset([quant_block], quant_val_inps, quant_val_inps, dev, attention_mask, position_ids, prefixed_key_values)
    
    layers_student[start_layer] = layers_student[start_layer].to("cpu")
    layers_teacher[start_layer] = layers_teacher[start_layer].to("cpu")
    
    try:
        torch.npu.empty_cache()
    except AttributeError:
        try:
            torch.cuda.empty_cache()
        except AttributeError:
            pass
    gc.collect()
    
    student_model.config.use_cache = use_cache
    teacher_model.config.use_cache = use_cache
    
    return {
        "layer_idx": layer_idx,
        "best_val_loss": best_val_loss.item(),
        "final_train_loss": train_loss_history[-1] if train_loss_history else None,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_model_path", type=str, help="Path to quantized student model")
    parser.add_argument("--fp_model_path", type=str, help="Path to full-precision teacher model")
    parser.add_argument("--model_name", type=str, default=None, help="Model name for saving data cache")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="Cache directory for dataset")
    parser.add_argument("--output_dir", default="./log/layer_analysis", type=str, help="Logging directory")
    
    parser.add_argument("--loss_type", type=str, default="mse", help="Loss type for hidden states: mse, clamp_mse, etc.")
    parser.add_argument("--hidden_weight", type=float, default=1.0, help="Weight for hidden states loss")
    parser.add_argument("--attn_weight", type=float, default=0.1, help="Weight for attention KL divergence loss")
    
    parser.add_argument("--quant_lr", type=float, default=5e-6, help="Learning rate for quantization parameters")
    parser.add_argument("--weight_lr", type=float, default=0.0, help="Learning rate for FP weights")
    parser.add_argument("--min_lr_factor", type=float, default=10, help="Min LR factor")
    parser.add_argument("--clip_grad", type=float, default=0.3)
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--use_fp32", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--early_stop", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--constant_wlr", action="store_true")
    
    parser.add_argument("--train_size", type=int, default=512, help="Number of calibration data samples")
    parser.add_argument("--val_size", type=int, default=64, help="Number of validation data samples")
    parser.add_argument("--training_seqlen", type=int, default=1024, help="Training sequence length")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per layer")
    parser.add_argument("--calib_dataset", type=str, default="pile",
                        choices=["wikitext2", "ptb", "c4", "mix", "redpajama", "pile"],
                        help="Calibration dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seed", type=int, default=2, help="Random seed")
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="Perplexity sequence length")
    
    parser.add_argument("--eval_ppl", action="store_true", default=True, help="Evaluate perplexity after each layer")
    parser.add_argument("--eval_tasks", type=str, default="", help="Evaluation tasks (not used in layer analysis)")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--max_memory", type=str, default="55GiB", help="Max memory per device")
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    try:
        torch.npu.manual_seed(args.seed)
    except AttributeError:
        pass
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    
    output_dir = Path(args.output_dir)
    logger = create_logger(output_dir)
    logger.info(args)
    
    if args.model_name is None:
        args.model_name = args.quant_model_path.split('/')[-1]
        logger.info(f"model_name is None, setting as {args.model_name}")
    
    quant_config = load_json_as_namespace(os.path.join(args.quant_model_path, 'prefixequant_config.json'))
    
    if quant_config.set_prefixed_tokens:
        prefixed_key_values = torch.load(os.path.join(args.quant_model_path, 'prefixed_key_values.pth'))
    else:
        prefixed_key_values = None
    
    logger.info("Loading student model (quantized)...")
    config = AutoConfig.from_pretrained(args.quant_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.quant_model_path, use_fast=False, legacy=False, trust_remote_code=True)
    
    with init_empty_weights():
        student_model = AutoModelForCausalLM.from_pretrained(
            args.quant_model_path, config=config, device_map='cpu', torch_dtype=torch.float16, trust_remote_code=True
        )
    wrap_to_quant_model(student_model)
    
    if quant_config.down_online_had:
        register_online_had(student_model)
    
    rope_function_name = model_utils.get_rope_function_name(student_model)
    layers = model_utils.get_layers(student_model)
    for layer in layers:
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
            layer.self_attn, 
            rope_function_name, 
            config=student_model.config,
            online_had=quant_config.qk_online_had
        )
    
    if quant_config.wbits < 16:
        logger.info('Initializing weight quantizer')
        init_weight_quantizer(quant_config, student_model, minmax_init=False)
    if quant_config.input_bits < 16:
        logger.info('Initializing input quantizer')
        init_input_quantizer(quant_config, student_model, minmax_init=False)
    if quant_config.v_bits < 16:
        logger.info('Initializing v quantizer')
        init_v_quantizer(quant_config, student_model, minmax_init=False)
    if quant_config.k_bits < 16:
        logger.info('Initializing k quantizer')
        init_k_quantizer(quant_config, student_model, minmax_init=False)
    
    device_map = infer_auto_device_map(student_model)
    logger.info("Loading quantized weights...")
    load_checkpoint_in_model(student_model, checkpoint=args.quant_model_path, device_map=device_map, dtype=torch.float16)
    student_model.half()
    
    logger.info("Loading teacher model (full-precision)...")
    dtype = torch.float16 if not args.use_fp32 else torch.float32
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.fp_model_path, device_map='cpu', torch_dtype=dtype, trust_remote_code=True
    )
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    pre_rotate = getattr(quant_config, 'pre_rotate', False)
    if pre_rotate:
        rotate_mode = getattr(quant_config, 'rotate_mode', 'hadamard')
        rotation_utils.fuse_layer_norms(teacher_model)
        rotation_utils.rotate_model(teacher_model, rotate_mode=rotate_mode, online=quant_config.down_online_had)
        teacher_model.half()
    
    trainloader, valloader = get_loaders(
        args.calib_dataset,
        tokenizer,
        args.train_size,
        args.val_size,
        seed=args.seed,
        seqlen=args.training_seqlen,
    )
    
    total_layers = len(model_utils.get_layers(student_model))
    logger.info(f"Total layers to analyze: {total_layers}")
    
    layer_results = []
    
    for layer_idx in range(total_layers):
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting analysis for layer {layer_idx}/{total_layers-1}")
        logger.info(f"{'='*60}\n")
        
        result = distill_single_layer(student_model, teacher_model, prefixed_key_values, layer_idx, args, trainloader, valloader, logger)
        layer_results.append(result)
        
        if args.eval_ppl:
            logger.info(f"\nEvaluating perplexity after layer {layer_idx}...")
            set_quant_state(student_model, weight_quant=True, act_quant=True)
            evaluate(student_model, tokenizer, prefixed_key_values, args, logger)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Layer {layer_idx} analysis complete")
        logger.info(f"  Best val loss: {result['best_val_loss']:.4f}")
        logger.info(f"  Final train loss: {result['final_train_loss']:.4f}")
        logger.info(f"{'='*60}\n")
    
    logger.info(f"\n{'='*60}")
    logger.info("Layer Analysis Summary")
    logger.info(f"{'='*60}")
    for result in layer_results:
        logger.info(f"Layer {result['layer_idx']:2d}: best_val_loss={result['best_val_loss']:.4f}, final_train_loss={result['final_train_loss']:.4f}")
    
    logger.info(f"\n{'='*60}")
    logger.info("Potential problematic layers (high val loss):")
    logger.info(f"{'='*60}")
    val_losses = [r['best_val_loss'] for r in layer_results]
    mean_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)
    threshold = mean_val_loss + 2 * std_val_loss
    
    for result in layer_results:
        if result['best_val_loss'] > threshold:
            logger.info(f"  Layer {result['layer_idx']:2d}: {result['best_val_loss']:.4f} (threshold: {threshold:.4f})")
    
    summary_file = output_dir / "layer_analysis_summary.json"
    import json
    with open(summary_file, 'w') as f:
        json.dump({
            "layer_results": layer_results,
            "mean_val_loss": float(mean_val_loss),
            "std_val_loss": float(std_val_loss),
            "threshold": float(threshold)
        }, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    print(sys.argv)
    main()
