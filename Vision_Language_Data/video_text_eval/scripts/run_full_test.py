#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video-Text 数据集评估

支持的数据集:
- 通用 JSONL 格式数据集

支持的指标:
- Frame Diversity: 帧多样性（光流）
- Semantic Diversity: 语义多样性（Inception V3）
- Object Consistency: 对象一致性（CLIP）
- Cross-Modal Consistency: 跨模态一致性（ViCLIP）
- Safety Bench: 安全性评估（GPT-4 Vision）
- Holistic Fidelity: 整体保真度（VBench）
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
# 添加父目录到 sys.path，以便导入上级模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime


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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输出目录
    output_dir = os.path.join(script_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'frame_diversity_results.json')
    
    print(f"\n{'='*70}")
    print(f"Frame Diversity 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输出目录
    output_dir = os.path.join(script_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'semantic_diversity_results.json')
    
    print(f"\n{'='*70}")
    print(f"Semantic Diversity 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输出目录
    output_dir = os.path.join(script_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'object_consistency_results.json')
    
    print(f"\n{'='*70}")
    print(f"Object Consistency 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输出目录
    output_dir = os.path.join(script_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'cross_modal_consistency_results.json')
    
    print(f"\n{'='*70}")
    print(f"Cross-Modal Consistency 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
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

def run_safety_bench(dataset_key: str, api_key: str = None, max_samples: int = None, dimensions: list = None):
    """运行安全性评估
    
    Args:
        dataset_key: 数据集标识
        api_key: OpenAI API Key
        max_samples: 最大样本数（用于测试）
        dimensions: 要评估的维度列表
    """
    if dataset_key not in DATASETS:
        print(f"未知数据集: {dataset_key}")
        print(f"可用数据集: {list(DATASETS.keys())}")
        return
    
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if not api_key:
        print("错误: 请设置 OPENAI_API_KEY 环境变量或通过 --api-key 参数提供")
        return
    
    config = DATASETS[dataset_key]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输出目录
    output_dir = os.path.join(script_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'safety_bench_results.json')
    
    print(f"\n{'='*70}")
    print(f"Safety Bench 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输出目录
    output_dir = os.path.join(script_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'holistic_fidelity_results.json')
    
    print(f"\n{'='*70}")
    print(f"Holistic Fidelity (VBench) 评估: {config['name']}")
    if max_samples:
        print(f"样本限制: {max_samples}")
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
# 全量评估
# =============================================================================

def run_all(dataset_key: str, max_samples: int = None, skip_safety: bool = False, skip_vbench: bool = False):
    """运行所有评估指标
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
        skip_safety: 是否跳过安全性评估（需要 API Key）
        skip_vbench: 是否跳过 VBench 评估（需要 VBench 依赖）
    """
    print(f"\n{'='*70}")
    print(f"开始全量评估: {dataset_key}")
    print(f"{'='*70}\n")
    
    all_results = {}
    
    # 1. Frame Diversity
    print("\n[1/6] Frame Diversity...")
    try:
        all_results['frame_diversity'] = run_frame_diversity(dataset_key, max_samples)
    except Exception as e:
        print(f"Frame Diversity 评估失败: {e}")
        all_results['frame_diversity'] = {'error': str(e)}
    
    # 2. Semantic Diversity
    print("\n[2/6] Semantic Diversity...")
    try:
        all_results['semantic_diversity'] = run_semantic_diversity(dataset_key, max_samples)
    except Exception as e:
        print(f"Semantic Diversity 评估失败: {e}")
        all_results['semantic_diversity'] = {'error': str(e)}
    
    # 3. Object Consistency
    print("\n[3/6] Object Consistency...")
    try:
        all_results['object_consistency'] = run_object_consistency(dataset_key, max_samples)
    except Exception as e:
        print(f"Object Consistency 评估失败: {e}")
        all_results['object_consistency'] = {'error': str(e)}
    
    # 4. Cross-Modal Consistency
    print("\n[4/6] Cross-Modal Consistency...")
    try:
        all_results['cross_modal_consistency'] = run_cross_modal_consistency(dataset_key, max_samples)
    except Exception as e:
        print(f"Cross-Modal Consistency 评估失败: {e}")
        all_results['cross_modal_consistency'] = {'error': str(e)}
    
    # 5. Safety Bench (可选)
    if not skip_safety:
        print("\n[5/6] Safety Bench...")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            try:
                all_results['safety_bench'] = run_safety_bench(dataset_key, api_key, max_samples)
            except Exception as e:
                print(f"Safety Bench 评估失败: {e}")
                all_results['safety_bench'] = {'error': str(e)}
        else:
            print("跳过 Safety Bench（未设置 OPENAI_API_KEY）")
            all_results['safety_bench'] = {'skipped': 'No API key'}
    else:
        print("\n[5/6] Safety Bench... (跳过)")
        all_results['safety_bench'] = {'skipped': 'User requested'}
    
    # 6. Holistic Fidelity (可选)
    if not skip_vbench:
        print("\n[6/6] Holistic Fidelity (VBench)...")
        try:
            all_results['holistic_fidelity'] = run_holistic_fidelity(dataset_key, max_samples)
        except Exception as e:
            print(f"Holistic Fidelity 评估失败: {e}")
            all_results['holistic_fidelity'] = {'error': str(e)}
    else:
        print("\n[6/6] Holistic Fidelity (VBench)... (跳过)")
        all_results['holistic_fidelity'] = {'skipped': 'User requested'}
    
    # 保存汇总结果
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = os.path.join(output_dir, f'full_evaluation_{timestamp}.json')
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)
    
    print(f"\n{'='*70}")
    print(f"全量评估完成！")
    print(f"汇总结果: {summary_file}")
    print(f"{'='*70}\n")
    
    return all_results


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Video-Text 数据集评估工具')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='评估命令')
    
    # frame_diversity 命令
    frame_div_parser = subparsers.add_parser('frame_diversity', help='帧多样性评估')
    frame_div_parser.add_argument('--dataset', '-d', type=str, required=True, choices=list(DATASETS.keys()), help='数据集')
    frame_div_parser.add_argument('--max-samples', '-n', type=int, default=None, help='最大样本数')
    
    # semantic_diversity 命令
    sem_div_parser = subparsers.add_parser('semantic_diversity', help='语义多样性评估')
    sem_div_parser.add_argument('--dataset', '-d', type=str, required=True, choices=list(DATASETS.keys()), help='数据集')
    sem_div_parser.add_argument('--max-samples', '-n', type=int, default=None, help='最大样本数')
    
    # object_consistency 命令
    obj_con_parser = subparsers.add_parser('object_consistency', help='对象一致性评估')
    obj_con_parser.add_argument('--dataset', '-d', type=str, required=True, choices=list(DATASETS.keys()), help='数据集')
    obj_con_parser.add_argument('--max-samples', '-n', type=int, default=None, help='最大样本数')
    
    # cross_modal_consistency 命令
    cmc_parser = subparsers.add_parser('cross_modal_consistency', help='跨模态一致性评估')
    cmc_parser.add_argument('--dataset', '-d', type=str, required=True, choices=list(DATASETS.keys()), help='数据集')
    cmc_parser.add_argument('--max-samples', '-n', type=int, default=None, help='最大样本数')
    
    # safety_bench 命令
    safety_parser = subparsers.add_parser('safety_bench', help='安全性评估')
    safety_parser.add_argument('--dataset', '-d', type=str, required=True, choices=list(DATASETS.keys()), help='数据集')
    safety_parser.add_argument('--api-key', type=str, default=None, help='OpenAI API Key')
    safety_parser.add_argument('--max-samples', '-n', type=int, default=None, help='最大样本数')
    safety_parser.add_argument('--dimensions', type=str, nargs='+', default=None, help='评估维度')
    
    # holistic_fidelity 命令
    vbench_parser = subparsers.add_parser('holistic_fidelity', help='整体保真度评估 (VBench)')
    vbench_parser.add_argument('--dataset', '-d', type=str, required=True, choices=list(DATASETS.keys()), help='数据集')
    vbench_parser.add_argument('--max-samples', '-n', type=int, default=None, help='最大样本数')
    vbench_parser.add_argument('--dimensions', type=str, nargs='+', default=None, help='评估维度')
    
    # all 命令
    all_parser = subparsers.add_parser('all', help='运行所有评估')
    all_parser.add_argument('--dataset', '-d', type=str, required=True, choices=list(DATASETS.keys()), help='数据集')
    all_parser.add_argument('--max-samples', '-n', type=int, default=None, help='最大样本数')
    all_parser.add_argument('--skip-safety', action='store_true', help='跳过安全性评估')
    all_parser.add_argument('--skip-vbench', action='store_true', help='跳过 VBench 评估')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # 执行对应命令
    if args.command == 'frame_diversity':
        run_frame_diversity(args.dataset, args.max_samples)
    elif args.command == 'semantic_diversity':
        run_semantic_diversity(args.dataset, args.max_samples)
    elif args.command == 'object_consistency':
        run_object_consistency(args.dataset, args.max_samples)
    elif args.command == 'cross_modal_consistency':
        run_cross_modal_consistency(args.dataset, args.max_samples)
    elif args.command == 'safety_bench':
        run_safety_bench(args.dataset, args.api_key, args.max_samples, args.dimensions)
    elif args.command == 'holistic_fidelity':
        run_holistic_fidelity(args.dataset, args.max_samples, args.dimensions)
    elif args.command == 'all':
        run_all(args.dataset, args.max_samples, args.skip_safety, args.skip_vbench)


if __name__ == '__main__':
    main()
