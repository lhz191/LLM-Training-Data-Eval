#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video-Text 数据集评估

支持的数据集:
- test: 测试数据集

支持的指标:
- frame_diversity: 帧多样性（光流）
- semantic_diversity: 语义多样性（Inception V3）
- object_consistency: 对象一致性（CLIP）
- cross_modal_consistency: 跨模态一致性（ViCLIP）
- safety_bench: 安全性评估（GPT-4 Vision）
- holistic_fidelity: 整体保真度（VBench）
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
    'test': {
        'name': 'Test Dataset',
        'data_path': '/mnt/petrelfs/liuhaoze/main/Vision_Language_Data/LLMDataBenchmark/Multimodal/data_utils/test.jsonl',
    },
    # 可以添加更多数据集配置
}


# =============================================================================
# Frame Diversity 评估
# =============================================================================

def run_frame_diversity(dataset_key: str, max_samples: int = None):
    """运行帧多样性评估
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
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
    output_file = os.path.join(output_dir, 'frame_diversity_results.json')
    
    print(f"\n{'='*70}")
    print(f"Frame Diversity 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from loaders import GeneralLoader
    from metrics.frame_diversity import compute_frame_diversity
    
    loader = GeneralLoader(config['data_path'])
    
    results = compute_frame_diversity(
        data_iterator=loader.iterate(),
        output_file=output_file,
        max_samples=max_samples,
    )
    
    print(f"\n结果已保存到: {output_file}")
    return results


# =============================================================================
# Semantic Diversity 评估
# =============================================================================

def run_semantic_diversity(dataset_key: str, max_samples: int = None):
    """运行语义多样性评估
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
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
    output_file = os.path.join(output_dir, 'semantic_diversity_results.json')
    
    print(f"\n{'='*70}")
    print(f"Semantic Diversity 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from loaders import GeneralLoader
    from metrics.semantic_diversity import compute_semantic_diversity
    
    loader = GeneralLoader(config['data_path'])
    
    results = compute_semantic_diversity(
        data_iterator=loader.iterate(),
        output_file=output_file,
        max_samples=max_samples,
    )
    
    print(f"\n结果已保存到: {output_file}")
    return results


# =============================================================================
# Object Consistency 评估
# =============================================================================

def run_object_consistency(dataset_key: str, max_samples: int = None):
    """运行对象一致性评估
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
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
    output_file = os.path.join(output_dir, 'object_consistency_results.json')
    
    print(f"\n{'='*70}")
    print(f"Object Consistency 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from loaders import GeneralLoader
    from metrics.object_consistency import compute_object_consistency
    
    loader = GeneralLoader(config['data_path'])
    
    results = compute_object_consistency(
        data_iterator=loader.iterate(),
        output_file=output_file,
        max_samples=max_samples,
    )
    
    print(f"\n结果已保存到: {output_file}")
    return results


# =============================================================================
# Cross-Modal Consistency 评估
# =============================================================================

def run_cross_modal_consistency(dataset_key: str, max_samples: int = None):
    """运行跨模态一致性评估
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
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
    output_file = os.path.join(output_dir, 'cross_modal_consistency_results.json')
    
    print(f"\n{'='*70}")
    print(f"Cross-Modal Consistency 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from loaders import GeneralLoader
    from metrics.cross_modal_consistency import compute_cross_modal_consistency
    
    loader = GeneralLoader(config['data_path'])
    
    results = compute_cross_modal_consistency(
        data_iterator=loader.iterate(),
        output_file=output_file,
        max_samples=max_samples,
    )
    
    print(f"\n结果已保存到: {output_file}")
    return results


# =============================================================================
# Safety Bench 评估
# =============================================================================

def run_safety_bench(dataset_key: str, max_samples: int = None, api_key: str = None, dimensions: list = None):
    """运行安全性评估
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
        api_key: OpenAI API Key
        dimensions: 要评估的维度列表
    """
    if dataset_key not in DATASETS:
        print(f"未知数据集: {dataset_key}")
        print(f"可用数据集: {list(DATASETS.keys())}")
        return
    
    # 获取 API Key
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误：需要 OpenAI API Key")
        print("请通过 --api-key 参数或 OPENAI_API_KEY 环境变量提供")
        return
    
    config = DATASETS[dataset_key]
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 输出目录
    output_dir = os.path.join(module_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'safety_bench_results.json')
    
    print(f"\n{'='*70}")
    print(f"Safety Bench 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    if dimensions:
        print(f"评估维度: {dimensions}")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from loaders import GeneralLoader
    from metrics.safety_bench import compute_safety_bench
    
    loader = GeneralLoader(config['data_path'])
    
    results = compute_safety_bench(
        data_iterator=loader.iterate(),
        api_key=api_key,
        dimensions=dimensions,
        output_file=output_file,
        max_samples=max_samples,
    )
    
    print(f"\n结果已保存到: {output_file}")
    return results


# =============================================================================
# Holistic Fidelity 评估 (VBench)
# =============================================================================

def run_holistic_fidelity(dataset_key: str, max_samples: int = None, dimensions: list = None):
    """运行整体保真度评估 (VBench)
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
        dimensions: 要评估的维度列表
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
    output_file = os.path.join(output_dir, 'holistic_fidelity_results.json')
    
    print(f"\n{'='*70}")
    print(f"Holistic Fidelity (VBench) 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    if dimensions:
        print(f"评估维度: {dimensions}")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from loaders import GeneralLoader
    from metrics.holistic_fidelity import compute_holistic_fidelity
    
    loader = GeneralLoader(config['data_path'])
    
    results = compute_holistic_fidelity(
        data_iterator=loader.iterate(),
        dimension_list=dimensions,
        output_file=output_file,
        max_samples=max_samples,
    )
    
    print(f"\n结果已保存到: {output_file}")
    return results


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Video-Text 数据集评估')
    parser.add_argument('--dataset', '-d', type=str, default='test',
                        choices=['all'] + list(DATASETS.keys()),
                        help='要验证的数据集 (默认: test)')
    parser.add_argument('--metric', '-m', type=str, default='frame_diversity',
                        choices=['frame_diversity', 'semantic_diversity', 'object_consistency',
                                 'cross_modal_consistency', 'safety_bench', 'holistic_fidelity', 'all'],
                        help='评估指标 (默认: frame_diversity)')
    
    # 通用参数
    parser.add_argument('--max-samples', type=int, default=None,
                        help='评估的样本数 (默认: None 表示全量)')
    
    # Safety Bench 参数
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API Key（也可通过 OPENAI_API_KEY 环境变量设置）')
    
    # VBench / Safety Bench 维度参数
    parser.add_argument('--dimensions', type=str, nargs='+', default=None,
                        help='评估维度（用于 safety_bench 和 holistic_fidelity）')
    
    args = parser.parse_args()
    
    datasets_to_run = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]
    
    for key in datasets_to_run:
        if args.metric in ['frame_diversity', 'all']:
            run_frame_diversity(key, max_samples=args.max_samples)
        
        if args.metric in ['semantic_diversity', 'all']:
            run_semantic_diversity(key, max_samples=args.max_samples)
        
        if args.metric in ['object_consistency', 'all']:
            run_object_consistency(key, max_samples=args.max_samples)
        
        if args.metric in ['cross_modal_consistency', 'all']:
            run_cross_modal_consistency(key, max_samples=args.max_samples)
        
        if args.metric in ['safety_bench', 'all']:
            run_safety_bench(key, max_samples=args.max_samples, api_key=args.api_key, dimensions=args.dimensions)
        
        if args.metric in ['holistic_fidelity', 'all']:
            run_holistic_fidelity(key, max_samples=args.max_samples, dimensions=args.dimensions)


if __name__ == '__main__':
    main()
