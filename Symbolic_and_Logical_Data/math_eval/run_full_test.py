#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学数据集 Validity + Faithfulness + Format Check + Diversity 验证

支持的数据集:
- OpenMathInstruct-1
- LILA (Math)

支持的指标:
- Validity: 代码执行验证
- Faithfulness: LLM-as-Judge 推理质量评估
- Reasoning Validity: LLM-as-Judge 推理逻辑正确性评估
- Format Check: 数据格式验证
- Diversity: 多样性评估 (Vendi Score / KNN)
"""
import sys
sys.set_int_max_str_digits(0)

import os
import argparse
import itertools
from datetime import datetime

from validity import compute_validity, compute_validity_parallel
from faithfulness import compute_faithfulness
from reasoning_validity import compute_reasoning_validity, compute_reasoning_validity_parallel
from format_check import compute_format_check, compute_format_check_parallel
from diversity import compute_diversity
from code_executor import get_comparator
from openmath_executor import (
    OpenMathCodeExtractor, OpenMathExecutor, OpenMathExecutorFast, 
    BoxedAnswerExtractor, DirectAnswerExtractor,
    OpenMathResultComparator, OpenMathFormatChecker
)
from lila_executor import (
    LILACodeExtractor, LILACodeExecutor,
    LILAResultComparator, LILAFormatChecker
)
from loaders import OpenMathInstructLoader, LILALoader


# =============================================================================
# 数据集配置
# =============================================================================

DATASETS = {
    'openmathinstruct': {
        'name': 'OpenMathInstruct-1',
        'data_path': '/mnt/petrelfs/liuhaoze/datasets/Symbolic_and_Logical_Data/OpenMathInstruct-1',
        'result_file': 'results/openmath/validity_results.json',
        'log_file': 'results/openmath/validity_details.log',
        'loader_class': OpenMathInstructLoader,
        'loader_kwargs': {'use_correct': True},
        'code_extractor': OpenMathCodeExtractor,
        'executor': OpenMathExecutorFast,  # Fast 执行器
        'answer_extractor': BoxedAnswerExtractor,
        'comparator': OpenMathResultComparator,
        'format_checker': OpenMathFormatChecker,
        'progress_interval': 1000,
        # 多样性配置
        'diversity_method': 'knn',  # 'knn' 或 'vendi'
        'diversity_sample_size': None,  # None 表示全量，数字表示采样
        'embedding_cache': 'embeddings/openmath_question.npy',
        'embedding_model': 'all-MiniLM-L6-v2',  # 或 'Qwen/Qwen3-Embedding-8B'
    },
    'lila': {
        'name': 'LILA-Math',
        'data_path': '/mnt/petrelfs/liuhaoze/datasets/Symbolic_and_Logical_Data/LILA/lila/multi/iid/train_math_only.json',
        'result_file': 'results/lila/validity_results.json',
        'log_file': 'results/lila/validity_details.log',
        'loader_class': LILALoader,
        'loader_kwargs': {},
        'code_extractor': LILACodeExtractor,
        'executor': LILACodeExecutor,
        'answer_extractor': DirectAnswerExtractor,
        'comparator': LILAResultComparator,
        'format_checker': LILAFormatChecker,
        'progress_interval': 10000,
        # 多样性配置
        'diversity_method': 'knn',  # 'knn' 或 'vendi'
        'diversity_sample_size': None,  # None 表示全量
        'embedding_cache': 'embeddings/lila_question.npy',
        'embedding_model': 'all-MiniLM-L6-v2',  # 或 'Qwen/Qwen3-Embedding-8B'
    },
}


# =============================================================================
# 日志写入
# =============================================================================

def write_detailed_log(results: dict, log_path: str, dataset_name: str):
    """写入详细的日志文件"""
    with open(log_path, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# {dataset_name} Validity Analysis\n")
        f.write(f"# Timestamp: {results['timestamp']}\n")
        f.write(f"# Elapsed: {results['elapsed_seconds']:.1f}s\n")
        f.write("=" * 80 + "\n\n")
        
        # Statistics
        f.write("## Statistics\n\n")
        total = results['total']
        with_code = results['with_code']
        no_code = results['no_code']
        
        f.write(f"Total Samples: {total:,}\n")
        f.write(f"  With Code: {with_code:,} ({with_code/total*100:.1f}%)\n")
        f.write(f"  No Code (Pure CoT): {no_code:,} ({no_code/total*100:.1f}%)\n\n")
        
        f.write("### Code Execution Verification (samples with code)\n")
        f.write(f"  ✅ Match: {results['code_matches']:,}\n")
        f.write(f"  ❌ Exec Error: {results['code_exec_errors']:,}\n")
        f.write(f"  ❌ Mismatch: {results['code_mismatches']:,}\n")
        f.write(f"  ⚠️  No Expected Output: {results['code_no_expected']:,}\n")
        f.write(f"  📊 Code Acc: {results['code_acc']:.4f} ({results['code_acc']:.2%})\n\n")
        
        if no_code > 0:
            f.write("### Answer Verification (samples without code)\n")
            f.write(f"  ✅ Match: {results['nl_matches']:,}\n")
            f.write(f"  ❌ Mismatch: {results['nl_mismatches']:,}\n")
            f.write(f"  ⚠️  No Answer Extracted: {results['nl_no_answer']:,}\n")
            f.write(f"  ⚠️  No Ground Truth: {results['nl_no_gt']:,}\n")
            f.write(f"  📊 NL Acc: {results['nl_acc']:.4f} ({results['nl_acc']:.2%})\n\n")
        
        f.write("### Overall Metrics\n")
        f.write(f"  📊 Overall Acc: {results['overall_acc']:.4f} ({results['overall_acc']:.2%})\n\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Code Error Samples (限制数量)
        code_error_samples = results['code_error_samples']
        f.write(f"## ❌ Code Execution Error Details (Total: {len(code_error_samples)})\n\n")
        
        max_samples = 100
        for i, sample in enumerate(code_error_samples[:max_samples]):
            f.write(f"### Code Error #{i+1}\n")
            f.write(f"Sample ID: {sample['sample_id']}\n")
            f.write(f"Ground Truth: {sample['ground_truth']}\n")
            f.write(f"Error: {sample['error']}\n\n")
            question = sample['question'][:500] + '...' if len(sample['question']) > 500 else sample['question']
            f.write(f"Question:\n{question}\n\n")
            if sample['code']:
                code = sample['code'][:1000] + '...' if len(sample['code']) > 1000 else sample['code']
                f.write(f"Code:\n```python\n{code}\n```\n\n")
            f.write("-" * 80 + "\n\n")
        
        if len(code_error_samples) > max_samples:
            f.write(f"... 还有 {len(code_error_samples) - max_samples} 个错误样本未显示 ...\n\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Code Mismatch Samples
        code_mismatch_samples = results['code_mismatch_samples']
        f.write(f"## ❌ Code Result Mismatch Details (Total: {len(code_mismatch_samples)})\n\n")
        
        for i, sample in enumerate(code_mismatch_samples[:max_samples]):
            f.write(f"### Code Mismatch #{i+1}\n")
            f.write(f"Sample ID: {sample['sample_id']}\n")
            f.write(f"Ground Truth: {sample['ground_truth']}\n\n")
            question = sample['question'][:500] + '...' if len(sample['question']) > 500 else sample['question']
            f.write(f"Question:\n{question}\n\n")
            if sample['code']:
                code = sample['code'][:1000] + '...' if len(sample['code']) > 1000 else sample['code']
                f.write(f"Code:\n```python\n{code}\n```\n\n")
            f.write(f"Expected Output:\n```\n{sample['expected']}\n```\n\n")
            f.write(f"Actual Output:\n```\n{sample['actual']}\n```\n\n")
            f.write("-" * 80 + "\n\n")
        
        if len(code_mismatch_samples) > max_samples:
            f.write(f"... 还有 {len(code_mismatch_samples) - max_samples} 个不匹配样本未显示 ...\n\n")
        
        f.write("=" * 80 + "\n\n")
        
        # NL Mismatch Samples
        nl_mismatch_samples = results['nl_mismatch_samples']
        if nl_mismatch_samples:
            f.write(f"## ❌ Answer Mismatch Details (Total: {len(nl_mismatch_samples)})\n\n")
            
            for i, sample in enumerate(nl_mismatch_samples[:max_samples]):
                f.write(f"### Answer Mismatch #{i+1}\n")
                f.write(f"Sample ID: {sample['sample_id']}\n")
                f.write(f"Extracted Answer: {sample['extracted_answer']}\n")
                f.write(f"Ground Truth: {sample['ground_truth']}\n\n")
                question = sample['question'][:500] + '...' if len(sample['question']) > 500 else sample['question']
                f.write(f"Question:\n{question}\n\n")
                solution = str(sample['solution'])[:1000] + '...' if len(str(sample['solution'])) > 1000 else sample['solution']
                f.write(f"Solution:\n{solution}\n\n")
                f.write("-" * 80 + "\n\n")
            
            if len(nl_mismatch_samples) > max_samples:
                f.write(f"... 还有 {len(nl_mismatch_samples) - max_samples} 个不匹配样本未显示 ...\n\n")
            
            f.write("=" * 80 + "\n\n")
        
        f.write("# End of Report\n")


# =============================================================================
# 主函数
# =============================================================================

def run_dataset(dataset_key: str, parallel: bool = False, num_workers: int = None):
    """运行指定数据集的验证
    
    Args:
        dataset_key: 数据集标识
        parallel: 是否使用多进程并行
        num_workers: 并行进程数（仅当 parallel=True 时有效）
    """
    if dataset_key not in DATASETS:
        print(f"未知数据集: {dataset_key}")
        print(f"可用数据集: {list(DATASETS.keys())}")
        return
    
    config = DATASETS[dataset_key]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, config['result_file'])
    log_file = os.path.join(script_dir, config['log_file'])
    
    print(f"\n{'='*70}")
    print(f"开始验证: {config['name']}")
    if parallel:
        print(f"模式: 多进程并行 (workers={num_workers or 'auto'})")
    else:
        print(f"模式: 单进程串行")
    print(f"{'='*70}\n")
    
    # 创建 loader
    loader = config['loader_class'](config['data_path'], **config['loader_kwargs'])
    
    if parallel:
        # 多进程并行模式
        results = compute_validity_parallel(
            data_iterator=loader.iterate(),
            code_extractor_class=config['code_extractor'],
            executor_class=config['executor'],
            answer_extractor_class=config['answer_extractor'],
            comparator_class=config['comparator'],
            output_file=output_file,
            progress_interval=config['progress_interval'],
            dataset_name=config['name'],
            num_workers=num_workers,
        )
    else:
        # 单进程串行模式
        results = compute_validity(
            data_iterator=loader.iterate(),
            code_extractor=config['code_extractor'](),
            executor=config['executor'](),
            answer_extractor=config['answer_extractor'](),
            comparator=config['comparator'](),
            output_file=output_file,
            progress_interval=config['progress_interval'],
            dataset_name=config['name']
    )
    
    # 写入详细日志
    write_detailed_log(results, log_file, config['name'])
    print(f"\n详细日志已保存到: {log_file}")
    
    # 简要打印
    print("\n" + "=" * 70)
    print(f"代码执行错误样本数: {len(results['code_error_samples'])}")
    print(f"代码结果不匹配样本数: {len(results['code_mismatch_samples'])}")
    if results['nl_mismatch_samples']:
        print(f"答案不匹配样本数: {len(results['nl_mismatch_samples'])}")
    print("=" * 70)
    
    # 打印前几个错误样本
    if results['code_error_samples']:
        print("\n代码执行错误预览 (前 5 个):")
        for sample in results['code_error_samples'][:5]:
            print(f"  - {sample['sample_id']}: {sample['error'][:60]}...")
    
    if results['code_mismatch_samples']:
        print("\n代码结果不匹配预览 (前 5 个):")
        for sample in results['code_mismatch_samples'][:5]:
            exp_preview = str(sample['expected'])[:30].replace('\n', ' ')
            act_preview = str(sample['actual'])[:30].replace('\n', ' ') if sample['actual'] else 'None'
            print(f"  - {sample['sample_id']}: expected='{exp_preview}...' actual='{act_preview}...'")
    
    if results['nl_mismatch_samples']:
        print("\n答案不匹配预览 (前 5 个):")
        for sample in results['nl_mismatch_samples'][:5]:
            print(f"  - {sample['sample_id']}: extracted='{sample['extracted_answer']}' gt='{sample['ground_truth']}'")
    
    return results


def run_faithfulness(dataset_key: str, max_samples: int = None):
    """运行指定数据集的 Faithfulness 评估"""
    if dataset_key not in DATASETS:
        print(f"未知数据集: {dataset_key}")
        return
    
    config = DATASETS[dataset_key]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, f"{dataset_key}_faithfulness_results.json")
    
    print(f"\n{'='*70}")
    print(f"Faithfulness 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 创建 loader
    loader = config['loader_class'](config['data_path'], **config['loader_kwargs'])
    
    # 如果有样本限制，使用 islice
    if max_samples:
        data_iter = itertools.islice(loader.iterate(), max_samples)
    else:
        data_iter = loader.iterate()
    
    # 运行评估
    results = compute_faithfulness(
        data_iterator=data_iter,
        output_file=output_file,
        progress_interval=1000,  # 全量时每 1000 条输出一次
        dataset_name=config['name'],
        max_samples=max_samples
    )
    
    return results


def run_reasoning_validity(dataset_key: str, max_samples: int = None):
    """运行指定数据集的 Reasoning Validity 评估"""
    if dataset_key not in DATASETS:
        print(f"未知数据集: {dataset_key}")
        return
    
    config = DATASETS[dataset_key]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 根据数据集类型选择输出目录
    if 'lila' in dataset_key:
        output_dir = os.path.join(script_dir, 'results', 'lila')
    elif 'openmath' in dataset_key:
        output_dir = os.path.join(script_dir, 'results', 'openmath')
    else:
        output_dir = os.path.join(script_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'reasoning_validity_results.json')
    
    print(f"\n{'='*70}")
    print(f"Reasoning Validity 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 创建 loader
    loader = config['loader_class'](config['data_path'], **config['loader_kwargs'])
    
    # 如果有样本限制，使用 islice
    if max_samples:
        data_iter = itertools.islice(loader.iterate(), max_samples)
    else:
        data_iter = loader.iterate()
    
    # 运行评估（使用并行版本）
    results = compute_reasoning_validity_parallel(
        data_iterator=data_iter,
        code_extractor=config['code_extractor'](),
        answer_extractor=config['answer_extractor'](),
        comparator_class=config['comparator'],  # 使用数据集特定的比较器
        output_file=output_file,
        progress_interval=100,
        dataset_name=config['name'],
        max_samples=max_samples,
        max_workers=32,  # 并行进程数
    )
    
    return results


def run_format_check(dataset_key: str, max_samples: int = None, parallel: bool = True, num_workers: int = None):
    """运行指定数据集的 Format Check 评估"""
    if dataset_key not in DATASETS:
        print(f"未知数据集: {dataset_key}")
        return
    
    config = DATASETS[dataset_key]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 根据数据集类型选择输出目录
    if 'lila' in dataset_key:
        output_dir = os.path.join(script_dir, 'results', 'lila')
    elif 'openmath' in dataset_key:
        output_dir = os.path.join(script_dir, 'results', 'openmath')
    else:
        output_dir = os.path.join(script_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'format_check_results.json')
    
    print(f"\n{'='*70}")
    print(f"Format Check 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 创建 loader
    loader = config['loader_class'](config['data_path'], **config['loader_kwargs'])
    
    # 如果有样本限制，使用 islice
    if max_samples:
        data_iter = itertools.islice(loader.iterate(), max_samples)
    else:
        data_iter = loader.iterate()
    
    # 运行评估
    if parallel:
        results = compute_format_check_parallel(
            data_iterator=data_iter,
            format_checker_class=config['format_checker'],
            dataset_name=config['name'],
            output_file=output_file,
            max_samples=max_samples,
            progress_interval=1000,
            max_workers=num_workers,
        )
    else:
        checker = config['format_checker']()
        results = compute_format_check(
            data_iterator=data_iter,
            format_checker=checker,
            dataset_name=config['name'],
            output_file=output_file,
            max_samples=max_samples,
            progress_interval=1000,
        )
    
    return results


def run_diversity(dataset_key: str, max_samples: int = None, method: str = None, 
                  sample_size: int = None, embedding_model: str = None,
                  embedding_batch_size: int = None, vendi_batch_size: int = None, num_gpus: int = None):
    """运行指定数据集的 Diversity 多样性评估
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
        method: 多样性计算方法 ('knn' 或 'vendi')，None 表示使用配置默认值
        sample_size: 采样大小，None 表示使用配置默认值
        embedding_model: Embedding 模型名称，None 表示使用配置默认值
        embedding_batch_size: Embedding 生成时的 batch 大小
        vendi_batch_size: Vendi Score 分 batch 计算的大小
        num_gpus: Vendi Score 多 GPU 并行计算时使用的 GPU 数量
    """
    if dataset_key not in DATASETS:
        print(f"未知数据集: {dataset_key}")
        return
    
    config = DATASETS[dataset_key]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 根据数据集类型选择输出目录
    if 'lila' in dataset_key:
        output_dir = os.path.join(script_dir, 'results', 'lila')
    elif 'openmath' in dataset_key:
        output_dir = os.path.join(script_dir, 'results', 'openmath')
    else:
        output_dir = os.path.join(script_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用参数或配置的默认值
    diversity_method = method or config.get('diversity_method', 'knn')
    diversity_sample_size = sample_size if sample_size is not None else config.get('diversity_sample_size')
    emb_model = embedding_model or config.get('embedding_model', 'all-MiniLM-L6-v2')
    
    # 生成包含模型名的 embedding 缓存路径
    model_short_name = emb_model.split('/')[-1] if '/' in emb_model else emb_model
    embedding_cache = os.path.join(script_dir, f'embeddings/{dataset_key}_question_{model_short_name}.npy')
    
    # 生成包含模型名的输出文件名
    # 处理模型名中的特殊字符（如 Qwen/Qwen3-Embedding-8B -> Qwen3-Embedding-8B）
    model_short_name = emb_model.split('/')[-1] if '/' in emb_model else emb_model
    output_file = os.path.join(output_dir, f'diversity_{diversity_method}_{model_short_name}_results.json')
    
    print(f"\n{'='*70}")
    print(f"Diversity 评估: {config['name']}")
    print(f"方法: {diversity_method}")
    print(f"Embedding 模型: {emb_model}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    if diversity_sample_size:
        print(f"采样大小: {diversity_sample_size}")
    print(f"{'='*70}\n")
    
    # 创建 loader
    loader = config['loader_class'](config['data_path'], **config['loader_kwargs'])
    
    # 如果有样本限制，使用 islice
    if max_samples:
        data_iter = itertools.islice(loader.iterate(), max_samples)
    else:
        data_iter = loader.iterate()
    
    # 运行评估
    results = compute_diversity(
        data_iterator=data_iter,
        dataset_name=config['name'],
        method=diversity_method,
        field='question',
        embedding_model=emb_model,
        embedding_cache_path=embedding_cache,
        sample_size=diversity_sample_size,
        output_file=output_file,
        max_samples=max_samples,
        embedding_batch_size=embedding_batch_size,
        vendi_batch_size=vendi_batch_size,
        num_gpus=num_gpus,
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='数学数据集 Validity + Faithfulness + Reasoning Validity + Format Check + Diversity 验证')
    parser.add_argument('--dataset', '-d', type=str, default='all',
                        choices=['all'] + list(DATASETS.keys()),
                        help='要验证的数据集 (默认: all)')
    parser.add_argument('--metric', '-m', type=str, default='validity',
                        choices=['validity', 'faithfulness', 'reasoning_validity', 'format_check', 'diversity', 'all'],
                        help='评估指标 (默认: validity)')
    parser.add_argument('--diversity-method', type=str, default=None,
                        choices=['knn', 'vendi'],
                        help='多样性计算方法 (默认: 使用配置)')
    parser.add_argument('--diversity-sample-size', type=int, default=None,
                        help='多样性计算采样大小 (默认: 使用配置)')
    parser.add_argument('--embedding-model', type=str, default=None,
                        help='Embedding 模型: all-MiniLM-L6-v2, all-mpnet-base-v2, Qwen/Qwen3-Embedding-8B')
    parser.add_argument('--embedding-batch-size', type=int, default=None,
                        help='Embedding 生成时的 batch 大小 (默认: 64，大模型如 8B 建议用 4-8)')
    parser.add_argument('--vendi-batch-size', type=int, default=None,
                        help='Vendi Score 分 batch 计算的 batch 大小，用于节省显存 (默认: None 表示不分 batch，建议值: 10000-15000)')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Vendi Score 多 GPU 并行计算时使用的 GPU 数量 (默认: None 表示自动检测)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='评估的样本数 (默认: None 表示全量)')
    parser.add_argument('--parallel', '-p', action='store_true',
                        help='使用多进程并行加速 (默认: False)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='并行进程数 (默认: 自动检测CPU核心数)')
    args = parser.parse_args()
    
    datasets_to_run = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]
    
    for key in datasets_to_run:
        if args.metric in ['validity', 'all']:
            run_dataset(key, parallel=args.parallel, num_workers=args.workers)
        
        if args.metric in ['faithfulness', 'all']:
            run_faithfulness(key, max_samples=args.max_samples)
        
        if args.metric in ['reasoning_validity', 'all']:
            run_reasoning_validity(key, max_samples=args.max_samples)
        
        if args.metric in ['format_check', 'all']:
            run_format_check(key, max_samples=args.max_samples, parallel=args.parallel, num_workers=args.workers)
        
        if args.metric in ['diversity', 'all']:
            run_diversity(key, max_samples=args.max_samples, method=args.diversity_method, 
                         sample_size=args.diversity_sample_size, embedding_model=args.embedding_model,
                         embedding_batch_size=args.embedding_batch_size,
                         vendi_batch_size=args.vendi_batch_size,
                         num_gpus=args.num_gpus)


if __name__ == '__main__':
    main()
