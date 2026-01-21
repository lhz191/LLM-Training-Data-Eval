#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Agent 数据集评估

支持的数据集:
- ToolBench
- xLAM-60k

支持的指标:
- Format Check: 格式检查
- Executability: 静态可执行性检查
- Dynamic Executability: 动态可执行性检查（需要 RapidAPI Key）
- Diversity: 多样性评估 (Vendi Score / KNN)
"""

# ============================================================================
# 必须在任何 import 之前设置，防止 OpenBLAS/MKL 线程过多导致崩溃
# ============================================================================
import os
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '32'

import sys
sys.set_int_max_str_digits(0)

# 添加父目录到 sys.path，以便导入上级模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import itertools
from datetime import datetime


# =============================================================================
# 数据集配置
# =============================================================================

DATASETS = {
    'toolbench': {
        'name': 'ToolBench',
        'data_path': '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/toolbench_official/toolllama_G123_dfs_train.json',
        'toolenv_path': '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/toolbench_official/toolenv/tools',
        # 多样性配置
        'diversity_method': 'knn',
        'diversity_sample_size': None,
        'embedding_model': 'all-MiniLM-L6-v2',
    },
    'xlam': {
        'name': 'xLAM-60k',
        'data_path': '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/xlam_60k.jsonl',
        'toolenv_path': None,
        # 多样性配置
        'diversity_method': 'knn',
        'diversity_sample_size': None,
        'embedding_model': 'all-MiniLM-L6-v2',
    },
}


# =============================================================================
# Format Check 评估
# =============================================================================

def run_format_check(dataset_key: str, max_samples: int = None, parallel: bool = False, workers: int = None):
    """运行指定数据集的 Format Check 格式检查
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
        parallel: 是否使用并行模式
        workers: 并行进程数
    """
    if dataset_key not in DATASETS:
        print(f"未知数据集: {dataset_key}")
        print(f"可用数据集: {list(DATASETS.keys())}")
        return
    
    config = DATASETS[dataset_key]
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 输出目录
    output_dir = os.path.join(module_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'format_check_results.json')
    
    print(f"\n{'='*70}")
    print(f"Format Check 评估: {config['name']}")
    print(f"模式: {'并行' if parallel else '串行'}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from metrics.format_check import compute_format_check, compute_format_check_parallel
    
    if dataset_key == 'toolbench':
        from loaders import ToolBenchLoader
        from toolbench_executor import ToolBenchFormatChecker
        
        loader = ToolBenchLoader(config['data_path'])
        checker_class = ToolBenchFormatChecker
        
    elif dataset_key == 'xlam':
        from loaders import XLAMLoader
        from xlam_executor import XLAMFormatChecker
        
        loader = XLAMLoader(config['data_path'])
        checker_class = XLAMFormatChecker
    
    # 运行评估
    if parallel:
        results = compute_format_check_parallel(
            data_iterator=loader.iterate(),
            format_checker_class=checker_class,
            dataset_name=config['name'],
            output_file=output_file,
            max_samples=max_samples,
            max_workers=workers,
        )
    else:
        checker = checker_class()
        results = compute_format_check(
            data_iterator=loader.iterate(),
            format_checker=checker,
            dataset_name=config['name'],
            output_file=output_file,
            max_samples=max_samples,
        )
    
    return results


# =============================================================================
# Executability 评估
# =============================================================================

def run_executability(dataset_key: str, max_samples: int = None, parallel: bool = False, workers: int = None):
    """运行指定数据集的 Executability 静态可执行性检查
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
        parallel: 是否使用并行模式
        workers: 并行进程数
    """
    if dataset_key not in DATASETS:
        print(f"未知数据集: {dataset_key}")
        print(f"可用数据集: {list(DATASETS.keys())}")
        return
    
    config = DATASETS[dataset_key]
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 输出目录
    output_dir = os.path.join(module_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'executability_results.json')
    
    print(f"\n{'='*70}")
    print(f"Executability 评估: {config['name']}")
    print(f"模式: {'并行' if parallel else '串行'}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from metrics.executability import compute_executability, compute_executability_parallel
    
    if dataset_key == 'toolbench':
        from loaders import ToolBenchLoader
        from toolbench_executor import ToolBenchExecutabilityChecker
        
        loader = ToolBenchLoader(config['data_path'])
        checker_class = ToolBenchExecutabilityChecker
        checker_kwargs = {'toolenv_path': config['toolenv_path']}
        
    elif dataset_key == 'xlam':
        from loaders import XLAMLoader
        from xlam_executor import XLAMExecutabilityChecker
        
        loader = XLAMLoader(config['data_path'])
        checker_class = XLAMExecutabilityChecker
        checker_kwargs = {}
    
    # 运行评估
    if parallel:
        results = compute_executability_parallel(
            data_iterator=loader.iterate(),
            executability_checker_class=checker_class,
            checker_kwargs=checker_kwargs,
            dataset_name=config['name'],
            output_file=output_file,
            max_samples=max_samples,
            max_workers=workers,
        )
    else:
        checker = checker_class(**checker_kwargs)
        results = compute_executability(
            data_iterator=loader.iterate(),
            executability_checker=checker,
            dataset_name=config['name'],
            output_file=output_file,
            max_samples=max_samples,
        )
    
    return results


# =============================================================================
# Dynamic Executability 评估
# =============================================================================

def run_dynamic_executability(dataset_key: str, max_samples: int = None, workers: int = 16,
                               rapidapi_key: str = None, timeout: int = 30):
    """运行指定数据集的 Dynamic Executability 动态可执行性检查
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
        workers: 并行线程数
        rapidapi_key: RapidAPI Key
        timeout: API 调用超时时间
    """
    if dataset_key not in DATASETS:
        print(f"未知数据集: {dataset_key}")
        print(f"可用数据集: {list(DATASETS.keys())}")
        return
    
    if dataset_key != 'toolbench':
        print(f"Dynamic Executability 目前只支持 ToolBench 数据集")
        return
    
    # 获取 RapidAPI Key
    rapidapi_key = rapidapi_key or os.environ.get('RAPIDAPI_KEY')
    if not rapidapi_key:
        print("错误：需要 RapidAPI Key")
        print("请通过 --rapidapi-key 参数或 RAPIDAPI_KEY 环境变量提供")
        return
    
    config = DATASETS[dataset_key]
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 输出目录
    output_dir = os.path.join(module_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'dynamic_executability_results.json')
    
    print(f"\n{'='*70}")
    print(f"Dynamic Executability 评估: {config['name']}")
    print(f"并行线程: {workers}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from metrics.dynamic_executability import compute_dynamic_executability
    from loaders import ToolBenchLoader
    from toolbench_executor import ToolBenchDynamicChecker
    
    loader = ToolBenchLoader(config['data_path'])
    
    # 使用缓存目录加速 API 映射加载
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, 'cache')
    
    checker = ToolBenchDynamicChecker(
        rapidapi_key=rapidapi_key,
        toolenv_path=config['toolenv_path'],
        cache_dir=cache_dir,
        timeout=timeout,
    )
    
    # 运行评估
    results = compute_dynamic_executability(
        data_iterator=loader.iterate(),
        dynamic_checker=checker,
        dataset_name=config['name'],
        output_file=output_file,
        max_samples=max_samples,
        max_workers=workers,
    )
    
    return results


# =============================================================================
# Diversity 评估
# =============================================================================

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
        print(f"可用数据集: {list(DATASETS.keys())}")
        return
    
    config = DATASETS[dataset_key]
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 输出目录
    output_dir = os.path.join(module_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用参数或配置的默认值
    diversity_method = method or config.get('diversity_method', 'knn')
    diversity_sample_size = sample_size if sample_size is not None else config.get('diversity_sample_size')
    emb_model = embedding_model or config.get('embedding_model', 'all-MiniLM-L6-v2')
    
    # 生成包含模型名的 embedding 缓存路径
    model_short_name = emb_model.split('/')[-1] if '/' in emb_model else emb_model
    embedding_cache = os.path.join(script_dir, f'embeddings/{dataset_key}_query_{model_short_name}.npy')
    
    # 生成包含模型名的输出文件名
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
    
    # 导入相关模块
    from metrics.diversity import compute_diversity
    
    if dataset_key == 'toolbench':
        from loaders import ToolBenchLoader
        loader = ToolBenchLoader(config['data_path'])
    elif dataset_key == 'xlam':
        from loaders import XLAMLoader
        loader = XLAMLoader(config['data_path'])
    
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
        field='query',
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


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='API Agent 数据集评估')
    parser.add_argument('--dataset', '-d', type=str, default='toolbench',
                        choices=['all'] + list(DATASETS.keys()),
                        help='要验证的数据集 (默认: toolbench)')
    parser.add_argument('--metric', '-m', type=str, default='format_check',
                        choices=['format_check', 'executability', 'dynamic_executability', 'diversity', 'all'],
                        help='评估指标 (默认: format_check)')
    
    # 通用参数
    parser.add_argument('--max-samples', type=int, default=None,
                        help='评估的样本数 (默认: None 表示全量)')
    parser.add_argument('--parallel', action='store_true',
                        help='使用并行模式（format_check, executability）')
    parser.add_argument('--workers', type=int, default=None,
                        help='并行进程/线程数')
    
    # Dynamic Executability 参数
    parser.add_argument('--rapidapi-key', type=str, default=None,
                        help='RapidAPI Key（也可通过 RAPIDAPI_KEY 环境变量设置）')
    parser.add_argument('--timeout', type=int, default=30,
                        help='API 调用超时时间（秒）')
    
    # Diversity 参数
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
                        help='Vendi Score 分 batch 计算的 batch 大小，用于节省显存 (默认: None 表示不分 batch，建议值: 5000-10000)')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Vendi Score 多 GPU 并行计算时使用的 GPU 数量 (默认: None 表示自动检测)')
    
    args = parser.parse_args()
    
    datasets_to_run = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]
    
    for key in datasets_to_run:
        if args.metric in ['format_check', 'all']:
            run_format_check(
                key, 
                max_samples=args.max_samples, 
                parallel=args.parallel,
                workers=args.workers,
            )
        
        if args.metric in ['executability', 'all']:
            run_executability(
                key, 
                max_samples=args.max_samples, 
                parallel=args.parallel,
                workers=args.workers,
            )
        
        if args.metric in ['dynamic_executability', 'all']:
            run_dynamic_executability(
                key, 
                max_samples=args.max_samples, 
                workers=args.workers or 16,
                rapidapi_key=args.rapidapi_key,
                timeout=args.timeout,
            )
        
        if args.metric in ['diversity', 'all']:
            run_diversity(
                key, 
                max_samples=args.max_samples, 
                method=args.diversity_method, 
                sample_size=args.diversity_sample_size, 
                embedding_model=args.embedding_model,
                embedding_batch_size=args.embedding_batch_size,
                vendi_batch_size=args.vendi_batch_size,
                num_gpus=args.num_gpus
            )


if __name__ == '__main__':
    main()
