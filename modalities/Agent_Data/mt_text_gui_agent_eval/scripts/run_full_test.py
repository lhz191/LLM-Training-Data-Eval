#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮 Text GUI Agent 数据集评估

支持的数据集:
- WebLINX (多轮)

支持的指标:
- format_check: Session 格式检查
- static_executability: Session 静态可执行性检查
- diversity: 多样性统计（展平后调用单轮）
- trustworthy: Session 级别安全评估 (AgentDoG)
- task_complexity: 任务复杂度（展平后调用单轮）

使用方式:
    python run_full_test.py --dataset weblinx --metric format_check
    python run_full_test.py --dataset weblinx --metric format_check --max-samples 10
    python run_full_test.py --dataset weblinx --metric all
"""

import os
import sys
import argparse
import json
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
_mt_eval_dir = os.path.dirname(_script_dir)
_agent_data_dir = os.path.dirname(_mt_eval_dir)

if _mt_eval_dir not in sys.path:
    sys.path.insert(0, _mt_eval_dir)
if _agent_data_dir not in sys.path:
    sys.path.insert(0, _agent_data_dir)

# =============================================================================
# 数据集配置
# =============================================================================

DATASETS = {
    'weblinx': {
        'name': 'WebLINX (Multi-Turn)',
        'data_path': '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/weblinx/chat_data/data/chat',
        'raw_data_path': '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/weblinx/raw_data',
        'split': 'train',
        'has_static': True,
        'action_mapping': {
            'click': 'click',
            'text_input': 'fill',
            'change': 'select_option',
            'submit': 'click',
            'scroll': 'scroll_into_view',
            'load': 'goto',
        },
        'skip_actions': {'say'},
    },
}


# =============================================================================
# Format Check 评估
# =============================================================================

def run_format_check(
    dataset_key: str,
    max_samples: int = None,
):
    """运行 Session 格式检查"""
    config = DATASETS[dataset_key]

    output_dir = os.path.join(_mt_eval_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'format_check_results.json')

    print(f"\n{'=' * 70}")
    print(f"Format Check (MT): {config['name']}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"Sample limit: {max_samples}")
    print(f"{'=' * 70}\n")

    from loaders import MultiTurnWebLINXLoader
    from executor.weblinx import WebLINXSessionFormatChecker

    loader = MultiTurnWebLINXLoader(config['data_path'], split=config['split'])
    checker = WebLINXSessionFormatChecker()

    results = []
    total_errors = 0
    total_warnings = 0

    for idx, session in enumerate(loader.iterate()):
        if max_samples and idx >= max_samples:
            break

        errors, warnings, stats = checker.check_session(session)
        total_errors += len(errors)
        total_warnings += len(warnings)

        results.append({
            'session_id': session.session_id,
            'num_rounds': len(session.rounds),
            'total_actions': session.total_actions,
            'errors': errors,
            'warnings': warnings,
            'stats': stats,
        })

        if (idx + 1) % 100 == 0:
            print(f"  Checked {idx + 1} sessions, {total_errors} errors, {total_warnings} warnings")

    n = len(results)
    error_rate = total_errors / n if n > 0 else 0
    clean_sessions = sum(1 for r in results if not r['errors'])

    summary = {
        'dataset': config['name'],
        'total_sessions': n,
        'clean_sessions': clean_sessions,
        'clean_rate': clean_sessions / n if n > 0 else 0,
        'total_errors': total_errors,
        'total_warnings': total_warnings,
        'avg_errors_per_session': error_rate,
    }

    print(f"\n{'=' * 70}")
    print(f"Results: {n} sessions checked")
    print(f"  Clean sessions: {clean_sessions}/{n} ({summary['clean_rate']:.1%})")
    print(f"  Total errors: {total_errors}")
    print(f"  Total warnings: {total_warnings}")
    print(f"{'=' * 70}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'details': results}, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")

    return summary


# =============================================================================
# Static Executability 评估
# =============================================================================

def run_static_executability(
    dataset_key: str,
    max_samples: int = None,
    show_browser: bool = False,
):
    """运行 Session 静态可执行性检查"""
    config = DATASETS[dataset_key]

    if not config.get('has_static', False):
        print(f"{config['name']} does not support static executability")
        return None

    output_dir = os.path.join(_mt_eval_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'static_executability_results.json')

    print(f"\n{'=' * 70}")
    print(f"Static Executability (MT): {config['name']}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"Sample limit: {max_samples}")
    print(f"{'=' * 70}\n")

    from loaders import MultiTurnWebLINXLoader
    from executor.weblinx import WebLINXSessionStaticChecker

    loader = MultiTurnWebLINXLoader(config['data_path'], split=config['split'])
    checker = WebLINXSessionStaticChecker(
        raw_data_path=config.get('raw_data_path'),
        headless=not show_browser,
    )

    results = []
    for idx, session in enumerate(loader.iterate()):
        if max_samples and idx >= max_samples:
            break

        errors, warnings, stats = checker.check_session(session)
        results.append({
            'session_id': session.session_id,
            'stats': stats,
            'errors': errors,
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'details': results}, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")

    return results


# =============================================================================
# Diversity 评估
# =============================================================================

def run_diversity(
    dataset_key: str,
    max_samples: int = None,
):
    """运行多样性统计（展平 Session 后调用单轮函数）"""
    config = DATASETS[dataset_key]

    output_dir = os.path.join(_mt_eval_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'diversity_results.json')

    print(f"\n{'=' * 70}")
    print(f"Diversity (MT): {config['name']}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"Sample limit: {max_samples}")
    print(f"{'=' * 70}\n")

    from metrics import compute_diversity_multiturn
    from loaders import MultiTurnWebLINXLoader

    loader = MultiTurnWebLINXLoader(config['data_path'], split=config['split'])

    results = compute_diversity_multiturn(
        session_iterator=loader.iterate(),
        dataset_name=config['name'],
        output_file=output_file,
        max_samples=max_samples,
        action_mapping=config.get('action_mapping'),
        skip_actions=config.get('skip_actions'),
    )

    return results


# =============================================================================
# Trustworthy 评估
# =============================================================================

def run_trustworthy(
    dataset_key: str,
    model_path: str = None,
    max_samples: int = None,
):
    """运行 Session 级别安全评估"""
    config = DATASETS[dataset_key]

    output_dir = os.path.join(_mt_eval_dir, 'results', dataset_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'trustworthy_results.json')

    print(f"\n{'=' * 70}")
    print(f"Trustworthy (MT): {config['name']}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"Sample limit: {max_samples}")
    print(f"{'=' * 70}\n")

    from metrics import compute_trustworthy
    from loaders import MultiTurnWebLINXLoader

    loader = MultiTurnWebLINXLoader(config['data_path'], split=config['split'])

    if not model_path:
        model_path = '/mnt/petrelfs/liuhaoze/models/AgentDoG-FG-Qwen3-4B'

    results = compute_trustworthy(
        session_iterator=loader.iterate(),
        model_path=model_path,
        max_samples=max_samples,
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")

    return results


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Turn Text GUI Agent Evaluation')
    parser.add_argument('--dataset', '-d', type=str, default='weblinx',
                        choices=list(DATASETS.keys()),
                        help='Dataset to evaluate (default: weblinx)')
    parser.add_argument('--metric', '-m', type=str, default='format_check',
                        choices=[
                            'format_check', 'static_executability',
                            'diversity', 'trustworthy', 'all',
                        ],
                        help='Metric to run (default: format_check)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max sessions to evaluate (default: all)')
    parser.add_argument('--show', action='store_true',
                        help='Show browser window (non-headless)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Override default data path')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Guard model path (for trustworthy)')

    args = parser.parse_args()

    if args.data_path and args.dataset in DATASETS:
        DATASETS[args.dataset]['data_path'] = args.data_path

    key = args.dataset

    if args.metric in ['format_check', 'all']:
        run_format_check(key, max_samples=args.max_samples)

    if args.metric in ['static_executability', 'all']:
        run_static_executability(key, max_samples=args.max_samples, show_browser=args.show)

    if args.metric in ['diversity', 'all']:
        run_diversity(key, max_samples=args.max_samples)

    if args.metric in ['trustworthy', 'all']:
        run_trustworthy(key, model_path=args.model_path, max_samples=args.max_samples)


if __name__ == '__main__':
    main()
