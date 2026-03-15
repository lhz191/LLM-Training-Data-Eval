#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-to-Report 数据集评估

支持的数据集:
- IU X-Ray:   医学影像报告生成（放射学胸片）
- ShareGPT4V: 通用图像描述（GPT-4V / ShareCaptioner 标注）

支持的指标:
- format_check:    格式检查（字段完整性、<image> token 匹配等）
- validity:        有效性检查（图片可读性 + LLM 多模态 Judge）
- report_quality:  报告质量（数据集特有的 LLM Judge 评估）
- duplication:     重复性检测（精确重复 / 近似重复 / 图像路径重复）
- diversity:       多样性评估（语义 / 表达 / 词汇 / 长度 / 来源分布）
"""

# ============================================================================
# 线程数限制
# ============================================================================
import os
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '32'

import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_module_dir = os.path.dirname(_script_dir)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_module_dir))))

if _module_dir not in sys.path:
    sys.path.insert(0, _module_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import itertools
from datetime import datetime


# =============================================================================
# 数据集配置
# =============================================================================

DATASETS = {
    'iu_xray': {
        'name': 'IU X-Ray',
        'data_path': '/mnt/petrelfs/liuhaoze/datasets/Multimodal_Data/IU-Xray',
        'image_base_dir': '/mnt/petrelfs/liuhaoze/datasets/Multimodal_Data/IU-Xray',
        'split': 'train',
        'loader': 'iu_xray',
        'format_checker': 'iu_xray',
        'quality_checker': 'iu_xray',
        'embedding_model': 'all-MiniLM-L6-v2',
        'diversity_method': 'knn',
    },
    'sharegpt4v': {
        'name': 'ShareGPT4V',
        'data_path': '/mnt/petrelfs/liuhaoze/datasets/Multimodal_Data/ShareGPT4V/sharegpt4v_report_all.json',
        'image_base_dir': '/mnt/petrelfs/liuhaoze/datasets/Multimodal_Data/ShareGPT4V/data',
        'loader': 'sharegpt4v',
        'format_checker': 'sharegpt4v',
        'quality_checker': 'sharegpt4v',
        'embedding_model': 'all-MiniLM-L6-v2',
        'diversity_method': 'knn',
    },
}


def _get_loader(dataset_key: str, config: dict):
    """根据配置创建 loader 实例"""
    if config['loader'] == 'iu_xray':
        from loaders import IUXRayLoader
        return IUXRayLoader(config['data_path'], split=config.get('split', 'train'))
    elif config['loader'] == 'sharegpt4v':
        from loaders import ShareGPT4VLoader
        return ShareGPT4VLoader(config['data_path'])
    else:
        raise ValueError(f"Unknown loader type: {config['loader']}")


def _get_format_checker(config: dict):
    """根据配置创建 FormatChecker 实例"""
    if config['format_checker'] == 'iu_xray':
        from executor import IUXRayFormatChecker
        return IUXRayFormatChecker()
    elif config['format_checker'] == 'sharegpt4v':
        from executor import ShareGPT4VFormatChecker
        return ShareGPT4VFormatChecker()
    else:
        from executor import GeneralFormatChecker
        return GeneralFormatChecker()


def _get_quality_checker(config: dict):
    """根据配置创建 QualityChecker 实例"""
    if config['quality_checker'] == 'iu_xray':
        from executor import IUXRayQualityChecker
        return IUXRayQualityChecker()
    elif config['quality_checker'] == 'sharegpt4v':
        from executor import ShareGPT4VQualityChecker
        return ShareGPT4VQualityChecker()
    else:
        raise ValueError(f"Unknown quality checker: {config['quality_checker']}")


def _print_header(metric_name: str, dataset_name: str, max_samples=None, parallel=False):
    print(f"\n{'='*70}")
    print(f"{metric_name}: {dataset_name}")
    print(f"模式: {'并行' if parallel else '串行'}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"样本: 全量")
    print(f"{'='*70}\n")


# =============================================================================
# Format Check
# =============================================================================

def run_format_check(dataset_key: str, max_samples: int = None, **kwargs):
    config = DATASETS[dataset_key]
    output_dir = os.path.join(_module_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'format_check_results.json')

    _print_header("Format Check", config['name'], max_samples)

    from metrics.format_check import compute_format_check

    loader = _get_loader(dataset_key, config)
    checker = _get_format_checker(config)

    return compute_format_check(
        data_iterator=loader.iterate(),
        format_checker=checker,
        dataset_name=config['name'],
        output_file=output_file,
        max_samples=max_samples,
    )


# =============================================================================
# Validity
# =============================================================================

def run_validity(dataset_key: str, max_samples: int = None, parallel: bool = False,
                 workers: int = 8, skip_llm: bool = False, **kwargs):
    config = DATASETS[dataset_key]
    output_dir = os.path.join(_module_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'validity_results.json')

    _print_header("Validity", config['name'], max_samples, parallel)

    from metrics.validity import compute_validity

    loader = _get_loader(dataset_key, config)

    return compute_validity(
        data_iterator=loader.iterate(),
        image_base_dir=config.get('image_base_dir'),
        dataset_name=config['name'],
        output_file=output_file,
        max_samples=max_samples,
        skip_llm=skip_llm,
        parallel=parallel,
        max_workers=workers,
    )


# =============================================================================
# Report Quality
# =============================================================================

def run_report_quality(dataset_key: str, max_samples: int = None,
                       parallel: bool = False, workers: int = 8, **kwargs):
    config = DATASETS[dataset_key]
    output_dir = os.path.join(_module_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'report_quality_results.json')

    _print_header("Report Quality", config['name'], max_samples, parallel)

    from metrics.report_quality import compute_report_quality

    loader = _get_loader(dataset_key, config)
    checker = _get_quality_checker(config)

    return compute_report_quality(
        data_iterator=loader.iterate(),
        quality_checker=checker,
        dataset_name=config['name'],
        output_file=output_file,
        max_samples=max_samples,
        parallel=parallel,
        max_workers=workers,
    )


# =============================================================================
# Duplication
# =============================================================================

def run_duplication(dataset_key: str, max_samples: int = None,
                    embedding_model: str = None, near_dup_threshold: float = 0.95, **kwargs):
    config = DATASETS[dataset_key]
    output_dir = os.path.join(_module_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'duplication_results.json')

    emb_model = embedding_model or config.get('embedding_model', 'all-MiniLM-L6-v2')
    model_short = emb_model.split('/')[-1] if '/' in emb_model else emb_model
    embedding_cache = os.path.join(_script_dir, f'embeddings/{dataset_key}_report_{model_short}.npy')

    _print_header("Duplication", config['name'], max_samples)

    from metrics.duplication import compute_duplication

    loader = _get_loader(dataset_key, config)

    return compute_duplication(
        data_iterator=loader.iterate(),
        dataset_name=config['name'],
        embedding_model=emb_model,
        embedding_cache_path=embedding_cache,
        near_dup_threshold=near_dup_threshold,
        output_file=output_file,
        max_samples=max_samples,
    )


# =============================================================================
# Diversity
# =============================================================================

def run_diversity(dataset_key: str, max_samples: int = None,
                  method: str = None, embedding_model: str = None,
                  sample_size: int = None, **kwargs):
    config = DATASETS[dataset_key]
    output_dir = os.path.join(_module_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)

    diversity_method = method or config.get('diversity_method', 'knn')
    emb_model = embedding_model or config.get('embedding_model', 'all-MiniLM-L6-v2')
    model_short = emb_model.split('/')[-1] if '/' in emb_model else emb_model
    embedding_cache = os.path.join(_script_dir, f'embeddings/{dataset_key}_report_{model_short}.npy')
    output_file = os.path.join(output_dir, f'diversity_{diversity_method}_{model_short}_results.json')

    _print_header("Diversity", config['name'], max_samples)
    print(f"方法: {diversity_method}")
    print(f"Embedding 模型: {emb_model}")
    if sample_size:
        print(f"采样大小: {sample_size}")
    print()

    from metrics.diversity import compute_diversity

    loader = _get_loader(dataset_key, config)

    return compute_diversity(
        data_iterator=loader.iterate(),
        dataset_name=config['name'],
        method=diversity_method,
        embedding_model=emb_model,
        embedding_cache_path=embedding_cache,
        sample_size=sample_size,
        output_file=output_file,
        max_samples=max_samples,
    )


# =============================================================================
# Main
# =============================================================================

METRIC_CHOICES = ['format_check', 'validity', 'report_quality', 'duplication', 'diversity', 'all']

METRIC_RUNNERS = {
    'format_check': run_format_check,
    'validity': run_validity,
    'report_quality': run_report_quality,
    'duplication': run_duplication,
    'diversity': run_diversity,
}


def main():
    parser = argparse.ArgumentParser(description='Image-to-Report 数据集评估')
    parser.add_argument('--dataset', '-d', type=str, default='iu_xray',
                        choices=['all'] + list(DATASETS.keys()),
                        help='数据集 (默认: iu_xray)')
    parser.add_argument('--metric', '-m', type=str, default='format_check',
                        choices=METRIC_CHOICES,
                        help='评估指标 (默认: format_check)')

    # 通用参数
    parser.add_argument('--max-samples', type=int, default=None,
                        help='最大样本数 (默认: None 全量)')
    parser.add_argument('--parallel', action='store_true',
                        help='并行模式 (validity, report_quality)')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行线程数 (默认: 8)')

    # 数据路径覆盖（方便跨集群使用）
    parser.add_argument('--data-path', type=str, default=None,
                        help='覆盖数据集路径')
    parser.add_argument('--image-base-dir', type=str, default=None,
                        help='覆盖图片根目录')

    # Validity 参数
    parser.add_argument('--skip-llm', action='store_true',
                        help='跳过 LLM Judge（仅检查图片可读性）')

    # Diversity 参数
    parser.add_argument('--diversity-method', type=str, default=None,
                        choices=['knn', 'vendi'],
                        help='多样性计算方法')
    parser.add_argument('--embedding-model', type=str, default=None,
                        help='Embedding 模型')
    parser.add_argument('--diversity-sample-size', type=int, default=None,
                        help='多样性计算采样大小')

    # Duplication 参数
    parser.add_argument('--near-dup-threshold', type=float, default=0.95,
                        help='近似重复阈值 (默认: 0.95)')

    args = parser.parse_args()

    # 路径覆盖
    if args.data_path:
        for key in DATASETS:
            DATASETS[key]['data_path'] = args.data_path
    if args.image_base_dir:
        for key in DATASETS:
            DATASETS[key]['image_base_dir'] = args.image_base_dir

    datasets_to_run = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]
    metrics_to_run = list(METRIC_RUNNERS.keys()) if args.metric == 'all' else [args.metric]

    common_kwargs = {
        'max_samples': args.max_samples,
        'parallel': args.parallel,
        'workers': args.workers,
        'skip_llm': args.skip_llm,
        'method': args.diversity_method,
        'embedding_model': args.embedding_model,
        'sample_size': args.diversity_sample_size,
        'near_dup_threshold': args.near_dup_threshold,
    }

    for dataset_key in datasets_to_run:
        if dataset_key not in DATASETS:
            print(f"未知数据集: {dataset_key}")
            continue

        for metric in metrics_to_run:
            runner = METRIC_RUNNERS[metric]
            runner(dataset_key, **common_kwargs)


if __name__ == '__main__':
    main()
