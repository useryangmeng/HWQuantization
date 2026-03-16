import os
import sys
import random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import utils.model_utils as model_utils
from utils.train_utils import create_logger

torch.backends.cudnn.benchmark = True

@torch.no_grad()
def test_ppl(args, model, tokenizer, prefixed_key_values=None, datasets=['wikitext2']):
    results = {}
    from utils.data_utils import get_loaders
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            tokenizer,
            seed=0,
            seqlen=args.ppl_seqlen,
            test_only=True
        )
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        seqlen = args.ppl_seqlen
        nsamples = testenc.numel() // seqlen
        model.eval()
        nlls = []
        from tqdm import tqdm
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)]
            labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)]
            batch = batch.to(model.device)
            labels = labels.to(model.device)
            
            if prefixed_key_values is not None:
                device_prefixed_key_values = tuple(
                    tuple(tensor.to(model.device) if tensor is not None else None 
                         for tensor in layer_kv) 
                    for layer_kv in prefixed_key_values
                )
            else:
                device_prefixed_key_values = None
                
            outputs = model(batch, labels=labels, past_key_values=device_prefixed_key_values)
            neg_log_likelihood = outputs.loss * seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        results[dataset] = ppl.item()
        print(f'{dataset}: {ppl}')
    return results

@torch.no_grad()
def evaluate(model, tokenizer, prefixed_key_values, args, logger):
    if prefixed_key_values is not None:
        prefixed_key_values = model_utils.mv_kv_cache(prefixed_key_values, model)
    
    results_str = ""
    
    # 执行PPL测试
    if args.eval_ppl:
        datasets = ["wikitext2", "c4"]
        ppl_results = test_ppl(args, model, tokenizer, prefixed_key_values, datasets)
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')
            results_str += f"{ppl_results[dataset]:.2f} "

    # 执行下游任务测试
    if args.eval_tasks:
        if prefixed_key_values is not None:
            model = model_utils.WrappedPrefixCausalLM(model, prefixed_key_values)
        
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        
        task_list = args.eval_tasks.split(',')
        model = HFLM(pretrained=model, batch_size=args.eval_batch_size)
        task_manager = lm_eval.tasks.TaskManager()

        
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=task_list,
            num_fewshot=0,
            task_manager=task_manager,
        )
        logger.info(make_table(results))
        
        total_acc = 0
        total_acc_with_norm = 0
        valid_tasks = 0
        
        for task in task_list:
            if task in results['results']:
                if 'acc,none' in results['results'][task]:
                    total_acc += results['results'][task]['acc,none']
                    results_str += f"{results['results'][task]['acc,none']*100:.2f} "
                    
                    if 'acc_norm,none' in results['results'][task]:
                        total_acc_with_norm += results['results'][task]['acc_norm,none']
                        results_str += f"{results['results'][task]['acc_norm,none']*100:.2f} "
                    else:
                        total_acc_with_norm += results['results'][task]['acc,none']
                    
                    valid_tasks += 1
        
        if valid_tasks > 0:
            logger.info(f'Average Acc: {total_acc/valid_tasks*100:.2f}%')
            logger.info(f'Average Acc (with norm): {total_acc_with_norm/valid_tasks*100:.2f}%')
        
        logger.info(f'Results string: {results_str.strip()}')
        
        if prefixed_key_values is not None:
            model = model.model

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="path to the model")
    parser.add_argument("--output_dir", default="./log/test/normal", type=str, help="direction of logging file")
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="length of the training sequence.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true", help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--eval_tasks", type=str, default="", help="example:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # 创建日志目录和日志器
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = create_logger(output_dir)
    
    logger.info(f"开始测试模型: {args.model_path}")
    logger.info(f"测试配置: PPL={args.eval_ppl}, 任务={args.eval_tasks}")

    # 检查是否有prefixed_key_values（可选）
    prefixed_key_values = None

    # 直接加载普通模型
    logger.info("加载模型和tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, 
            use_fast=False, 
            legacy=False, 
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info(f"模型加载成功，设备映射: {model.hf_device_map}")
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return

    # 执行测试
    logger.info("开始执行测试...")
    evaluate(model, tokenizer, prefixed_key_values, args, logger)
    logger.info("测试完成")

if __name__ == "__main__":
    print("hello")
    main()