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
        # 本机路径
        'data_path': '/home/liuhaoze/Desktop/mind2web',
        'raw_dump_path': '/home/liuhaoze/data/raw_dump',
        # 远程路径（集群）
        # 'data_path': '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/Mind2Web/data',
        # 'raw_dump_path': '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/Mind2Web/raw_dump',
        'has_dynamic': True,  # 支持动态可执行性（真实网站）
    },
    'webshop': {
        'name': 'WebShop',
        'data_path': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'webshop/baseline_models/data/il_trajs_finalized_images.jsonl'),
        'use_browser': False,  # 默认使用 Text 环境
        'has_dynamic': False,  # 仿真环境，不支持动态可执行性
    },
    'weblinx': {
        'name': 'WebLINX',
        # 本机路径 - data_dir 是目录，loader 会自动拼接 {split}.json.gz
        'data_path': '/home/liuhaoze/Desktop/mind2web/weblinx',
        # 远程路径（集群）
        # 'data_path': '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/weblinx/chat_data/data/chat',
        'has_dynamic': True,  # 支持动态可执行性（真实网站）
    },
}


# =============================================================================
# Static Executability 评估
# =============================================================================

def run_static_executability(
    dataset_key: str, 
    max_samples: int = None, 
    show_browser: bool = False,
    progress_interval: int = 10,
    use_browser: bool = False,  # WebShop: 使用 Browser 环境
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
    
    # WebShop 区分 Text/Browser 环境的结果文件
    if dataset_key == 'webshop':
        env_suffix = '_browser' if use_browser else '_text'
        output_file = os.path.join(output_dir, f'static_executability{env_suffix}_results.json')
    else:
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
        
        # 加载数据 - 先完整解析所有数据
        loader = Mind2WebLoader(config['data_path'])
        print("正在解析 Mind2Web 数据...")
        all_records = loader.parse_all(show_progress=True)
        print(f"解析完成，共 {len(all_records)} 条记录")
        print()
        
        # 创建检查器
        checker = Mind2WebStaticChecker(
            raw_dump_path=config['raw_dump_path'],
            headless=not show_browser,
        )
        
    elif dataset_key == 'webshop':
        from loaders import WebShopLoader
        from webshop_executor import WebShopStaticChecker
        
        # 加载数据 - 先完整解析所有数据
        loader = WebShopLoader(config['data_path'])
        print("正在解析 WebShop 数据...")
        all_records = loader.parse_all(show_progress=True)
        print(f"解析完成，共 {len(all_records)} 条记录")
        print()
        
        # 创建检查器
        # use_browser: 来自命令行 --browser 参数
        # show_browser: 来自命令行 --show 参数，控制是否显示浏览器窗口
        checker = WebShopStaticChecker(
            use_browser=use_browser,
            render=show_browser,  # 是否显示浏览器窗口
        )
        
        env_type = "Browser" if use_browser else "Text"
        print(f"WebShop 环境: {env_type}")
        
    elif dataset_key == 'weblinx':
        # TODO: WebLINX checker 还未实现
        raise NotImplementedError("WebLINX static checker not implemented yet")
    
    # 运行评估
    # mind2web 和 webshop 都使用预先解析好的 all_records
    if dataset_key in ('mind2web', 'webshop'):
        data_iter = iter(all_records)
    else:
        data_iter = loader.iterate()
    
    results = compute_static_executability(
        data_iterator=data_iter,
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
    
    # 检查是否支持动态可执行性
    if not config.get('has_dynamic', False):
        print(f"\n⚠ {config['name']} 不支持 dynamic executability")
        print(f"  请使用 static_executability 指标")
        return None
    
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
        
        # 加载数据 - 先完整解析所有数据
        loader = Mind2WebLoader(config['data_path'])
        print("正在解析 Mind2Web 数据...")
        all_records = loader.parse_all(show_progress=True)
        print(f"解析完成，共 {len(all_records)} 条记录")
        print()
        
        # 创建检查器
        checker = Mind2WebDynamicChecker(
            headless=not show_browser,
        )
        
    elif dataset_key == 'weblinx':
        # TODO: WebLINX checker 还未实现
        raise NotImplementedError("WebLINX dynamic checker not implemented yet")
    
    # 运行评估 - 使用预先解析好的 all_records
    results = compute_dynamic_executability(
        data_iterator=iter(all_records),
        dynamic_checker=checker,
        dataset_name=config['name'],
        output_file=output_file,
        max_samples=max_samples,
        progress_interval=progress_interval,
        execute=execute,
    )
    
    return results


# =============================================================================
# Format Check 评估
# =============================================================================

def run_format_check(
    dataset_key: str, 
    max_samples: int = None, 
    progress_interval: int = 100,
):
    """运行指定数据集的 Format Check 格式检查
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
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
    output_file = os.path.join(output_dir, 'format_check_results.json')
    
    print(f"\n{'='*70}")
    print(f"Format Check 评估: {config['name']}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from metrics.format_check import compute_format_check
    
    if dataset_key == 'mind2web':
        from loaders import Mind2WebLoader
        from mind2web_executor import Mind2WebFormatChecker
        
        # 加载数据 - 先完整解析所有数据
        loader = Mind2WebLoader(config['data_path'])
        print("正在解析 Mind2Web 数据...")
        all_records = loader.parse_all(show_progress=True)
        print(f"解析完成，共 {len(all_records)} 条记录")
        print()
        
        # 创建检查器
        checker = Mind2WebFormatChecker()
        
    elif dataset_key == 'webshop':
        from loaders import WebShopLoader
        from webshop_executor import WebShopFormatChecker
        
        # 加载数据 - 先完整解析所有数据
        loader = WebShopLoader(config['data_path'])
        print("正在解析 WebShop 数据...")
        all_records = loader.parse_all(show_progress=True)
        print(f"解析完成，共 {len(all_records)} 条记录")
        print()
        
        # 创建检查器
        checker = WebShopFormatChecker()
    
    elif dataset_key == 'weblinx':
        from loaders import WebLINXLoader
        from weblinx_executor import WebLINXFormatChecker
        
        # 加载数据 - 先完整解析所有数据
        loader = WebLINXLoader(config['data_path'], 'train')
        print("正在解析 WebLINX 数据...")
        all_records = loader.parse_all(show_progress=True)
        print(f"解析完成，共 {len(all_records)} 条记录")
        print()
        
        # 创建检查器
        checker = WebLINXFormatChecker()
    
    # 运行评估
    results = compute_format_check(
        data_iterator=iter(all_records),
        format_checker=checker,
        dataset_name=config['name'],
        output_file=output_file,
        max_samples=max_samples,
        progress_interval=progress_interval,
    )
    
    return results


# =============================================================================
# HTML Retention 评估
# =============================================================================

def run_html_retention(
    dataset_key: str, 
    max_samples: int = None, 
    progress_interval: int = 100,
):
    """运行指定数据集的 HTML 信息保留率检查
    
    Args:
        dataset_key: 数据集标识
        max_samples: 最大样本数（用于测试）
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
    output_file = os.path.join(output_dir, 'html_retention_results.json')
    
    print(f"\n{'='*70}")
    print(f"HTML Retention 评估: {config['name']}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    else:
        print(f"模式: 全量")
    print(f"{'='*70}\n")
    
    # 导入相关模块
    from metrics.html_retention import compute_html_retention
    from text_gui_executor import get_html_locator
    
    if dataset_key == 'mind2web':
        from loaders import Mind2WebLoader
        
        # 加载数据
        loader = Mind2WebLoader(config['data_path'])
        print("正在解析 Mind2Web 数据...")
        all_records = loader.parse_all(show_progress=True)
        print(f"解析完成，共 {len(all_records)} 条记录")
        print()
        
    elif dataset_key == 'webshop':
        from loaders import WebShopLoader
        
        # 加载数据
        loader = WebShopLoader(config['data_path'])
        print("正在解析 WebShop 数据...")
        all_records = loader.parse_all(show_progress=True)
        print(f"解析完成，共 {len(all_records)} 条记录")
        print()
    
    elif dataset_key == 'weblinx':
        from loaders import WebLINXLoader
        
        # 加载数据
        loader = WebLINXLoader(config['data_path'], 'train')
        print("正在解析 WebLINX 数据...")
        all_records = loader.parse_all(show_progress=True)
        print(f"解析完成，共 {len(all_records)} 条记录")
        print()
    
    # 限制样本数
    if max_samples:
        all_records = all_records[:max_samples]
    
    # 获取定位器
    locator = get_html_locator(dataset_key)
    
    # 运行评估
    results = compute_html_retention(
        records=all_records,
        locator=locator,
        output_file=output_file,
        show_progress=True,
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
                        choices=['static_executability', 'dynamic_executability', 'format_check', 'html_retention', 'all'],
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
    
    # WebShop 特有参数
    parser.add_argument('--browser', action='store_true',
                        help='WebShop: 使用浏览器环境而非 Text 环境（需要先启动 Flask 服务器）')
    
    args = parser.parse_args()
    
    # 覆盖配置
    if args.data_path and args.dataset in DATASETS:
        DATASETS[args.dataset]['data_path'] = args.data_path
    if args.raw_dump and args.dataset in DATASETS:
        DATASETS[args.dataset]['raw_dump_path'] = args.raw_dump
    
    datasets_to_run = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]
    
    for key in datasets_to_run:
        if args.metric in ['format_check', 'all']:
            run_format_check(
                key, 
                max_samples=args.max_samples,
                progress_interval=args.progress_interval,
            )
        
        if args.metric in ['static_executability', 'all']:
            run_static_executability(
                key, 
                max_samples=args.max_samples,
                show_browser=args.show,
                progress_interval=args.progress_interval,
                use_browser=args.browser,  # WebShop: Browser 环境
            )
        
        if args.metric in ['dynamic_executability', 'all']:
            run_dynamic_executability(
                key, 
                max_samples=args.max_samples,
                show_browser=args.show,
                progress_interval=args.progress_interval if args.metric == 'dynamic_executability' else 1,
                execute=not args.no_execute,
            )
        
        if args.metric in ['html_retention', 'all']:
            run_html_retention(
                key, 
                max_samples=args.max_samples,
                progress_interval=args.progress_interval,
            )


if __name__ == '__main__':
    main()
