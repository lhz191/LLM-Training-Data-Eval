#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Training Data Evaluation - 统一评估入口

统一的命令行入口，可以调用任意模态的评估。

Usage:
    # 格式检查
    python -m common.evaluate --modality api --dataset toolbench --metric format
    python -m common.evaluate --modality gui --dataset mind2web --metric format
    python -m common.evaluate --modality math --dataset lila --metric format
    
    # 可执行性检查
    python -m common.evaluate --modality api --dataset toolbench --metric executability
    python -m common.evaluate --modality gui --dataset mind2web --metric static
    
    # 动态检查
    python -m common.evaluate --modality api --dataset toolbench --metric dynamic
    python -m common.evaluate --modality gui --dataset mind2web --metric dynamic
    
    # Math 特有
    python -m common.evaluate --modality math --dataset lila --metric validity
    
    # 列出所有可用选项
    python -m common.evaluate --list
"""

import sys
import os
import argparse
import importlib
from typing import Optional

# 添加项目根目录到 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# 模态到模块的映射
# =============================================================================

MODALITY_MODULES = {
    'api': 'modalities.Agent_Data.api_agent_eval',
    'gui': 'modalities.Agent_Data.text_gui_agent_eval',
    'math': 'modalities.Symbolic_and_Logical_Data.math_eval',
    'image': 'modalities.Vision_Language_Data.image_text_eval',
    'video': 'modalities.Vision_Language_Data.video_text_eval',
}

MODALITY_DESCRIPTIONS = {
    'api': 'API Agent (ToolBench, xLAM)',
    'gui': 'GUI Agent (Mind2Web, WebShop, WebLINX)',
    'math': 'Math/Symbolic (LILA, OpenMath)',
    'image': 'Image-Text',
    'video': 'Video-Text',
}


# =============================================================================
# 主函数
# =============================================================================

def list_all():
    """列出所有可用的模态和数据集"""
    print("=" * 60)
    print("LLM Training Data Evaluation - Available Options")
    print("=" * 60)
    
    for modality, desc in MODALITY_DESCRIPTIONS.items():
        module_path = MODALITY_MODULES[modality]
        print(f"\n📁 {modality}: {desc}")
        print(f"   Module: {module_path}")
        
        # 尝试导入并列出数据集
        try:
            # 尝试导入 scripts/run_full_test.py 获取 DATASETS
            run_module = importlib.import_module(f'{module_path}.scripts.run_full_test')
            if hasattr(run_module, 'DATASETS'):
                datasets = list(run_module.DATASETS.keys())
                print(f"   Datasets: {', '.join(datasets)}")
        except (ImportError, ModuleNotFoundError):
            print(f"   Datasets: (module not found)")


def run_evaluation(modality: str, dataset: str, metric: str, **kwargs):
    """
    运行评估
    
    Args:
        modality: 模态
        dataset: 数据集
        metric: 评估指标
        **kwargs: 其他参数
    """
    if modality not in MODALITY_MODULES:
        print(f"❌ Unknown modality: {modality}")
        print(f"   Available: {', '.join(MODALITY_MODULES.keys())}")
        sys.exit(1)
    
    module_path = MODALITY_MODULES[modality]
    
    print("=" * 60)
    print(f"Running: {modality} / {dataset} / {metric}")
    print("=" * 60)
    
    try:
        # 导入对应模态的 run_full_test 模块
        run_module = importlib.import_module(f'{module_path}.scripts.run_full_test')
        
        # 构建参数
        args = argparse.Namespace(
            dataset=dataset,
            metric=metric,
            **kwargs
        )
        
        # 调用 main 函数（如果存在）
        if hasattr(run_module, 'main'):
            run_module.main(args)
        else:
            print(f"⚠️ Module {module_path}.scripts.run_full_test has no 'main' function")
            print(f"   Please run directly: python -m {module_path}.scripts.run_full_test --help")
            
    except (ImportError, ModuleNotFoundError) as e:
        print(f"❌ Failed to import module: {module_path}")
        print(f"   Error: {e}")
        print(f"\n   Try running directly:")
        print(f"   cd {module_path.replace('.', '/')}")
        print(f"   python scripts/run_full_test.py --dataset {dataset} --metric {metric}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='LLM Training Data Evaluation - 统一入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 列出所有可用选项
    python -m common.evaluate --list
    
    # API Agent 格式检查
    python -m common.evaluate -m api -d toolbench --metric format
    
    # GUI Agent 静态可执行性
    python -m common.evaluate -m gui -d mind2web --metric static
    
    # Math 代码执行验证
    python -m common.evaluate -m math -d lila --metric validity
        """
    )
    
    parser.add_argument('--list', '-l', action='store_true',
                        help='列出所有可用的模态和数据集')
    parser.add_argument('--modality', '-m', type=str,
                        choices=list(MODALITY_MODULES.keys()),
                        help='模态: api, gui, math, image, video')
    parser.add_argument('--dataset', '-d', type=str,
                        help='数据集名称')
    parser.add_argument('--metric', type=str,
                        help='评估指标')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='最大样本数')
    parser.add_argument('--parallel', '-p', action='store_true',
                        help='使用并行模式')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='并行进程数')
    
    args = parser.parse_args()
    
    if args.list:
        list_all()
        return
    
    if not args.modality:
        parser.print_help()
        print("\n❌ Please specify --modality (-m)")
        sys.exit(1)
    
    if not args.dataset:
        parser.print_help()
        print("\n❌ Please specify --dataset (-d)")
        sys.exit(1)
    
    if not args.metric:
        parser.print_help()
        print("\n❌ Please specify --metric")
        sys.exit(1)
    
    # 构建 kwargs
    kwargs = {
        'max_samples': args.max_samples,
        'parallel': args.parallel,
        'workers': args.workers,
    }
    # 移除 None 值
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    run_evaluation(args.modality, args.dataset, args.metric, **kwargs)


if __name__ == '__main__':
    main()
