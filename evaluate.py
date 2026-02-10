#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Training Data Evaluation - 统一评估入口

一个统一的命令行工具，可以评估任意模态的训练数据质量。

支持的模态:
- api: API Agent (ToolBench, xLAM)
- gui: GUI Agent (Mind2Web, WebShop, WebLINX)
- math: Math/Symbolic (LILA, OpenMath)
- image: Image-Text
- video: Video-Text

Usage:
    # 查看帮助
    python evaluate.py --help
    
    # 列出所有可用选项
    python evaluate.py --list
    
    # API Agent 评测
    python evaluate.py api toolbench format
    python evaluate.py api toolbench executability
    
    # GUI Agent 评测
    python evaluate.py gui mind2web format
    python evaluate.py gui mind2web static
    python evaluate.py gui weblinx static --max-samples 100
    
    # Math 评测
    python evaluate.py math lila format
    python evaluate.py math lila validity
    
    # 通用参数
    python evaluate.py <modality> <dataset> <metric> [--max-samples N] [--parallel]
"""

import sys
import os
import argparse
import importlib

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# 模态配置
# =============================================================================

MODALITIES = {
    'api': {
        'name': 'API Agent',
        'module': 'modalities.Agent_Data.api_agent_eval.scripts.run_full_test',
        'datasets': ['toolbench', 'xlam'],
        'metrics': ['format_check', 'executability', 'dynamic_executability', 'diversity', 'trustworthy'],
    },
    'gui': {
        'name': 'GUI Agent',
        'module': 'modalities.Agent_Data.text_gui_agent_eval.scripts.run_full_test',
        'datasets': ['mind2web', 'webshop', 'weblinx'],
        'metrics': ['format_check', 'static_executability', 'dynamic_executability', 'html_retention', 'diversity', 'trajectory_validity', 'task_complexity'],
    },
    'math': {
        'name': 'Math/Symbolic',
        'module': 'modalities.Symbolic_and_Logical_Data.math_eval.scripts.run_full_test',
        'datasets': ['lila', 'openmathinstruct'],
        'metrics': ['format_check', 'validity', 'faithfulness', 'reasoning_validity', 'diversity'],
    },
    'image': {
        'name': 'Image-Text',
        'module': 'modalities.Vision_Language_Data.image_text_eval.scripts.run_full_test',
        'datasets': ['coco'],
        'metrics': ['format_check', 'well_formed_rate', 'prompt_fidelity'],
    },
    'video': {
        'name': 'Video-Text',
        'module': 'modalities.Vision_Language_Data.video_text_eval.scripts.run_full_test',
        'datasets': [],
        'metrics': ['holistic_fidelity', 'semantic_diversity', 'safety_bench'],
    },
}


# =============================================================================
# 辅助函数
# =============================================================================

def print_banner():
    """打印横幅"""
    print("=" * 70)
    print("  LLM Training Data Evaluation Framework")
    print("  统一的训练数据质量评测工具")
    print("=" * 70)


def list_all():
    """列出所有可用选项"""
    print_banner()
    print()
    
    for key, config in MODALITIES.items():
        print(f"📁 {key}: {config['name']}")
        print(f"   Datasets: {', '.join(config['datasets']) if config['datasets'] else '(none)'}")
        print(f"   Metrics:  {', '.join(config['metrics'])}")
        print()
    
    print("-" * 70)
    print("Usage: python evaluate.py <modality> <dataset> <metric> [options]")
    print()
    print("Examples:")
    print("  python evaluate.py api toolbench format_check")
    print("  python evaluate.py gui mind2web static --max-samples 100")
    print("  python evaluate.py math lila validity --parallel")


def run_evaluation(modality: str, dataset: str, metric: str, args, extra_args=None):
    """运行评测
    
    Args:
        modality: 模态名称
        dataset: 数据集名称
        metric: 指标名称
        args: 已解析的通用参数
        extra_args: 未解析的额外参数，透传给子脚本
    """
    extra_args = extra_args or []
    if modality not in MODALITIES:
        print(f"❌ Unknown modality: {modality}")
        print(f"   Available: {', '.join(MODALITIES.keys())}")
        sys.exit(1)
    
    config = MODALITIES[modality]
    
    print_banner()
    print()
    print(f"  Modality: {config['name']} ({modality})")
    print(f"  Dataset:  {dataset}")
    print(f"  Metric:   {metric}")
    if args.max_samples:
        print(f"  Samples:  {args.max_samples}")
    if extra_args:
        print(f"  Extra:    {' '.join(extra_args)}")
    print()
    print("-" * 70)
    print()
    
    # 构建命令行参数
    cmd_args = [
        '--dataset', dataset,
        '--metric', metric,
    ]
    
    if args.max_samples:
        cmd_args.extend(['--max-samples', str(args.max_samples)])
    if args.parallel:
        cmd_args.append('--parallel')
    if args.workers:
        cmd_args.extend(['--workers', str(args.workers)])
    if args.show:
        cmd_args.append('--show')
    
    # 添加透传的额外参数
    cmd_args.extend(extra_args)
    
    # 动态导入并执行
    try:
        # 切换到对应模块目录
        module_parts = config['module'].rsplit('.', 1)
        module_dir = os.path.join(PROJECT_ROOT, module_parts[0].replace('.', os.sep))
        
        # 保存原始 argv 并替换
        original_argv = sys.argv
        sys.argv = ['run_full_test.py'] + cmd_args
        
        # 导入模块
        module = importlib.import_module(config['module'])
        
        # 调用模块的 main() 函数
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"⚠️ Module {config['module']} does not have a main() function")
        
        # 恢复 argv
        sys.argv = original_argv
        
    except ImportError as e:
        print(f"❌ Failed to import module: {config['module']}")
        print(f"   Error: {e}")
        print()
        print("   Try running directly:")
        print(f"   cd {config['module'].rsplit('.', 2)[0].replace('.', '/')}")
        print(f"   python scripts/run_full_test.py --dataset {dataset} --metric {metric}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LLM Training Data Evaluation - 统一评测入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate.py --list                           # 列出所有选项
    python evaluate.py api toolbench format_check       # API Agent 格式检查
    python evaluate.py gui mind2web static              # GUI Agent 静态检查
    python evaluate.py math lila validity               # Math 代码执行验证
    python evaluate.py gui weblinx static --max-samples 100 --show
        """
    )
    
    parser.add_argument('--list', '-l', action='store_true',
                        help='列出所有可用的模态、数据集和指标')
    parser.add_argument('modality', nargs='?', type=str,
                        help='模态: api, gui, math, image, video')
    parser.add_argument('dataset', nargs='?', type=str,
                        help='数据集名称')
    parser.add_argument('metric', nargs='?', type=str,
                        help='评测指标')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='最大样本数')
    parser.add_argument('--parallel', '-p', action='store_true',
                        help='使用并行模式')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='并行进程数')
    parser.add_argument('--show', action='store_true',
                        help='显示浏览器（GUI Agent）')
    
    # 使用 parse_known_args 捕获未知参数，透传给子脚本
    args, extra_args = parser.parse_known_args()
    
    # 列出所有选项
    if args.list or (not args.modality):
        list_all()
        return
    
    # 检查必需参数
    if not args.dataset:
        print("❌ Missing dataset. Usage: python evaluate.py <modality> <dataset> <metric>")
        sys.exit(1)
    
    if not args.metric:
        print("❌ Missing metric. Usage: python evaluate.py <modality> <dataset> <metric>")
        sys.exit(1)
    
    # 运行评测（extra_args 透传给子脚本）
    run_evaluation(args.modality, args.dataset, args.metric, args, extra_args)


if __name__ == '__main__':
    main()
