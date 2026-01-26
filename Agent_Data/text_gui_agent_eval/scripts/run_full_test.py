#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text GUI Agent 数据集评估

支持的数据集:
- Mind2Web

支持的指标:
- Static Executability: 静态可执行性检查（坐标定位 + 属性定位）
- Dynamic Executability: 动态可执行性检查（真实网站验证 + 执行）
"""

import os
import sys

# 添加父目录到 sys.path，以便导入上级模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime


# =============================================================================
# 数据集配置
# =============================================================================

DATASETS = {
    'mind2web': {
        'name': 'Mind2Web',
        'data_path': '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/Mind2Web/data',
        'raw_dump_path': '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/Mind2Web/raw_dump',
    },
    # 'weblinx': {
    #     'name': 'WebLINX',
    #     'data_path': '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/weblinx/chat_data',
    #     'raw_data_path': '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/weblinx/raw_data',
    # },
}


# =============================================================================
# Static Executability 评估
# =============================================================================

def run_static_executability(
    dataset_key: str, 
    max_samples: int = None, 
    show_browser: bool = False,
    progress_interval: int = 10,
):
    """运行指定数据集的 Static Executability 静态可执行性检查
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
        show_browser: 是否显示浏览器窗口
        progress_interval: 进度显示间隔
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
    output_file = os.path.join(output_dir, 'static_executability_results.json')
    
    print(f"\n{'='*70}")
    print(f"Static Executability 评估: {config['name']}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"浏览器: {'显示' if show_browser else '无头模式'}")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from metrics.static_executability import compute_static_executability
    
    if dataset_key == 'mind2web':
        from loaders import Mind2WebLoader
        from mind2web_executor import Mind2WebStaticChecker
        
        # 加载数据
        loader = Mind2WebLoader(config['data_path'])
        loader.load()
        
        # 创建检查器
        checker = Mind2WebStaticChecker(
            raw_dump_path=config['raw_dump_path'],
            headless=not show_browser,
        )
        
    elif dataset_key == 'weblinx':
        # TODO: WebLINX checker 还未实现
        raise NotImplementedError("WebLINX static checker not implemented yet")
    
    # 运行评估
    results = compute_static_executability(
        data_iterator=loader.iterate(),
        static_checker=checker,
        dataset_name=config['name'],
        output_file=output_file,
        max_samples=max_samples,
        progress_interval=progress_interval,
    )
    
    return results


# =============================================================================
# Dynamic Executability 评估
# =============================================================================

def run_dynamic_executability(
    dataset_key: str, 
    max_samples: int = None, 
    show_browser: bool = False,
    progress_interval: int = 1,
    execute: bool = True,
):
    """运行指定数据集的 Dynamic Executability 动态可执行性检查
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
        show_browser: 是否显示浏览器窗口
        progress_interval: 进度显示间隔
        execute: 是否执行操作（默认 True）
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
    output_file = os.path.join(output_dir, 'dynamic_executability_results.json')
    
    print(f"\n{'='*70}")
    print(f"Dynamic Executability 评估: {config['name']}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"浏览器: {'显示' if show_browser else '无头模式'}")
    print(f"执行模式: {'执行操作' if execute else '仅验证'}")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from metrics.dynamic_executability import compute_dynamic_executability
    
    if dataset_key == 'mind2web':
        from loaders import Mind2WebLoader
        from mind2web_executor import Mind2WebDynamicChecker
        
        # 加载数据
        loader = Mind2WebLoader(config['data_path'])
        loader.load()
        
        # 创建检查器
        checker = Mind2WebDynamicChecker(
            headless=not show_browser,
        )
        
    elif dataset_key == 'weblinx':
        # TODO: WebLINX checker 还未实现
        raise NotImplementedError("WebLINX dynamic checker not implemented yet")
    
    # 运行评估
    results = compute_dynamic_executability(
        data_iterator=loader.iterate(),
        dynamic_checker=checker,
        dataset_name=config['name'],
        output_file=output_file,
        max_samples=max_samples,
        progress_interval=progress_interval,
        execute=execute,
    )
    
    return results


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Text GUI Agent 数据集评估')
    parser.add_argument('--dataset', '-d', type=str, default='mind2web',
                        choices=['all'] + list(DATASETS.keys()),
                        help='要验证的数据集 (默认: mind2web)')
    parser.add_argument('--metric', '-m', type=str, default='static_executability',
                        choices=['static_executability', 'dynamic_executability', 'all'],
                        help='评估指标 (默认: static_executability)')
    
    # 通用参数
    parser.add_argument('--max-samples', type=int, default=None,
                        help='评估的样本数 (默认: None 表示全量)')
    parser.add_argument('--progress-interval', type=int, default=10,
                        help='进度显示间隔 (默认: 10)')
    
    # 浏览器参数
    parser.add_argument('--show', action='store_true',
                        help='显示浏览器窗口（而非无头模式）')
    
    # Static Executability 参数
    parser.add_argument('--raw-dump', type=str, default=None,
                        help='覆盖默认的 raw_dump 路径')
    parser.add_argument('--data-path', type=str, default=None,
                        help='覆盖默认的数据路径')
    
    # Dynamic Executability 参数
    parser.add_argument('--no-execute', action='store_true',
                        help='仅验证不执行操作（动态模式）')
    
    args = parser.parse_args()
    
    # 覆盖配置
    if args.data_path and args.dataset in DATASETS:
        DATASETS[args.dataset]['data_path'] = args.data_path
    if args.raw_dump and args.dataset in DATASETS:
        DATASETS[args.dataset]['raw_dump_path'] = args.raw_dump
    
    datasets_to_run = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]
    
    for key in datasets_to_run:
        if args.metric in ['static_executability', 'all']:
            run_static_executability(
                key, 
                max_samples=args.max_samples,
                show_browser=args.show,
                progress_interval=args.progress_interval,
            )
        
        if args.metric in ['dynamic_executability', 'all']:
            run_dynamic_executability(
                key, 
                max_samples=args.max_samples,
                show_browser=args.show,
                progress_interval=args.progress_interval if args.metric == 'dynamic_executability' else 1,
                execute=not args.no_execute,
            )


if __name__ == '__main__':
    main()
