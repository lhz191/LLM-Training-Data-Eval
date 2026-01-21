#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-Text 数据集评估

支持的数据集:
- coco_caption: COCO Caption 数据集

支持的指标:
- inception_score: Inception Score
- prompt_fidelity: Prompt Fidelity (CLIP)
- well_formed_rate: 格式正确率
- c2pa_validation: C2PA 验证率
"""

import os
import sys
# 添加父目录到 sys.path，以便导入上级模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse


# =============================================================================
# 数据集配置
# =============================================================================

DATASETS = {
    'coco_caption': {
        'name': 'COCO Caption',
        'data_path': '/mnt/petrelfs/liuhaoze/main/Vision_Language_Data/LLMDataBenchmark/Multimodal/data_utils/coco_caption.jsonl',
        'format_checker': 'coco_caption',
    },
    # 可以添加更多数据集配置
}


# =============================================================================
# Inception Score 评估
# =============================================================================

def run_inception_score(dataset_key: str, max_samples: int = None, batch_size: int = 64, device: str = 'cuda'):
    """运行 Inception Score 评估
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
        batch_size: 批次大小
        device: 设备
    """
    if dataset_key not in DATASETS:
        print(f"未知数据集: {dataset_key}")
        print(f"可用数据集: {list(DATASETS.keys())}")
        return
    
    config = DATASETS[dataset_key]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输出目录
    output_dir = os.path.join(script_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'inception_score_results.json')
    
    print(f"\n{'='*70}")
    print(f"Inception Score 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from loaders import GeneralLoader
    from metrics.inception_score import compute_inception_score
    import itertools
    
    loader = GeneralLoader(config['data_path'])
    
    # 如果有样本限制，使用 islice
    if max_samples:
        data_iter = itertools.islice(loader.iterate(), max_samples)
    else:
        data_iter = loader.iterate()
    
    score = compute_inception_score(
        data_iterator=data_iter,
        batch_size=batch_size,
        device=device,
    )
    
    # 保存结果
    import json
    results = {'inception_score': score}
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Inception Score: {score:.4f}")
    print(f"\n结果已保存到: {output_file}")
    return results


# =============================================================================
# Prompt Fidelity 评估
# =============================================================================

def run_prompt_fidelity(dataset_key: str, max_samples: int = None, batch_size: int = 64, device: str = 'cuda'):
    """运行 Prompt Fidelity 评估
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
        batch_size: 批次大小
        device: 设备
    """
    if dataset_key not in DATASETS:
        print(f"未知数据集: {dataset_key}")
        print(f"可用数据集: {list(DATASETS.keys())}")
        return
    
    config = DATASETS[dataset_key]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输出目录
    output_dir = os.path.join(script_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'prompt_fidelity_results.json')
    
    print(f"\n{'='*70}")
    print(f"Prompt Fidelity 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from loaders import GeneralLoader
    from metrics.prompt_fidelity import compute_prompt_fidelity
    import itertools
    
    loader = GeneralLoader(config['data_path'])
    
    # 如果有样本限制，使用 islice
    if max_samples:
        data_iter = itertools.islice(loader.iterate(), max_samples)
    else:
        data_iter = loader.iterate()
    
    score = compute_prompt_fidelity(
        data_iterator=data_iter,
        device=device,
        batch_size=batch_size,
    )
    
    # 保存结果
    import json
    results = {'prompt_fidelity': score}
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Prompt Fidelity: {score:.4f}")
    print(f"\n结果已保存到: {output_file}")
    return results


# =============================================================================
# Well-Formed Rate 评估
# =============================================================================

def run_well_formed_rate(dataset_key: str, max_samples: int = None, strict: bool = True):
    """运行 Well-Formed Rate 评估
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
        strict: 是否使用严格模式
    """
    if dataset_key not in DATASETS:
        print(f"未知数据集: {dataset_key}")
        print(f"可用数据集: {list(DATASETS.keys())}")
        return
    
    config = DATASETS[dataset_key]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输出目录
    output_dir = os.path.join(script_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'well_formed_rate_results.json')
    
    print(f"\n{'='*70}")
    print(f"Well-Formed Rate 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from loaders import GeneralLoader
    from metrics.well_formed_rate import compute_well_formed_rate
    from image_executor import get_format_checker
    import itertools
    
    loader = GeneralLoader(config['data_path'])
    checker = get_format_checker(config['format_checker'], strict=strict)
    
    # 如果有样本限制，使用 islice
    if max_samples:
        data_iter = itertools.islice(loader.iterate(), max_samples)
    else:
        data_iter = loader.iterate()
    
    score = compute_well_formed_rate(
        data_iterator=data_iter,
        checker=checker,
    )
    
    # 保存结果
    import json
    results = {'well_formed_rate': score}
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Well-Formed Rate: {score:.4f}")
    print(f"\n结果已保存到: {output_file}")
    return results


# =============================================================================
# C2PA Validation 评估
# =============================================================================

def run_c2pa_validation(dataset_key: str, max_samples: int = None, c2pa_tool: str = './c2patool'):
    """运行 C2PA Validation 评估
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
        c2pa_tool: c2patool 路径
    """
    if dataset_key not in DATASETS:
        print(f"未知数据集: {dataset_key}")
        print(f"可用数据集: {list(DATASETS.keys())}")
        return
    
    config = DATASETS[dataset_key]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输出目录
    output_dir = os.path.join(script_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'c2pa_validation_results.json')
    
    print(f"\n{'='*70}")
    print(f"C2PA Validation 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from loaders import GeneralLoader
    from metrics.c2pa_validation import compute_c2pa_validation_rate
    import itertools
    
    loader = GeneralLoader(config['data_path'])
    
    # 如果有样本限制，使用 islice
    if max_samples:
        data_iter = itertools.islice(loader.iterate(), max_samples)
    else:
        data_iter = loader.iterate()
    
    score = compute_c2pa_validation_rate(
        data_iterator=data_iter,
        tool_path=c2pa_tool,
    )
    
    # 保存结果
    import json
    results = {'c2pa_validation_rate': score}
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    if score is not None:
        print(f"C2PA Validation Rate: {score:.4f}")
    else:
        print("C2PA Validation Rate: N/A (no valid images)")
    print(f"\n结果已保存到: {output_file}")
    return results


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Image-Text 数据集评估')
    parser.add_argument('--dataset', '-d', type=str, default='coco_caption',
                        choices=['all'] + list(DATASETS.keys()),
                        help='要验证的数据集 (默认: coco_caption)')
    parser.add_argument('--metric', '-m', type=str, default='inception_score',
                        choices=['inception_score', 'prompt_fidelity', 'well_formed_rate', 'c2pa_validation', 'all'],
                        help='评估指标 (默认: inception_score)')
    
    # 通用参数
    parser.add_argument('--max-samples', type=int, default=None,
                        help='评估的样本数 (默认: None 表示全量)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小 (默认: 64)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (默认: cuda)')
    
    # Well-Formed Rate 参数
    parser.add_argument('--strict', action='store_true', default=True,
                        help='使用严格模式检查格式')
    
    # C2PA 参数
    parser.add_argument('--c2pa-tool', type=str, default='./c2patool',
                        help='c2patool 路径')
    
    args = parser.parse_args()
    
    datasets_to_run = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]
    
    for key in datasets_to_run:
        if args.metric in ['inception_score', 'all']:
            run_inception_score(key, max_samples=args.max_samples, batch_size=args.batch_size, device=args.device)
        
        if args.metric in ['prompt_fidelity', 'all']:
            run_prompt_fidelity(key, max_samples=args.max_samples, batch_size=args.batch_size, device=args.device)
        
        if args.metric in ['well_formed_rate', 'all']:
            run_well_formed_rate(key, max_samples=args.max_samples, strict=args.strict)
        
        if args.metric in ['c2pa_validation', 'all']:
            run_c2pa_validation(key, max_samples=args.max_samples, c2pa_tool=args.c2pa_tool)


if __name__ == '__main__':
    main()
