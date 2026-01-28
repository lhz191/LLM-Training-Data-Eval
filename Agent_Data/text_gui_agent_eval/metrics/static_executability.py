#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static Executability 指标 - GUI Agent 静态可执行性检查

通用框架，支持不同数据集的静态可执行性检查：
- 通用指标：total_records, total_actions, errors, warnings
- 数据集特有指标：由各 checker 返回，自动合并到结果中

使用方式:
    from metrics import compute_static_executability
    from loaders import Mind2WebLoader
    from mind2web_executor import Mind2WebStaticChecker
    
    loader = Mind2WebLoader('/path/to/mind2web/data')
    checker = Mind2WebStaticChecker(raw_dump_path='/path/to/raw_dump')
    
    results = compute_static_executability(
        data_iterator=loader.iterate(),
        static_checker=checker,
        dataset_name='Mind2Web',
        output_file='static_executability_results.json'
    )
"""

import json
import time
from datetime import datetime
from typing import Optional, Iterator, Dict, List, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_types import Record
from text_gui_executor import StaticExecutabilityChecker


def compute_static_executability(
    data_iterator: Iterator[Record],
    static_checker: StaticExecutabilityChecker,
    dataset_name: str = "unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 10,
) -> Dict[str, Any]:
    """
    计算静态可执行性指标
    
    Args:
        data_iterator: Record 迭代器
        static_checker: 静态可执行性检查器
        dataset_name: 数据集名称
        output_file: 结果输出文件
        max_samples: 最大样本数（用于测试）
        progress_interval: 进度显示间隔
    
    Returns:
        结果字典，包含：
        - 通用指标：total_records, total_actions, errors, warnings
        - 数据集特有指标：由 checker 返回，自动累加
    """
    print("=" * 70)
    print("Static Executability Evaluation")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    # =========================================================================
    # 通用统计（所有数据集共有）
    # =========================================================================
    total_records = 0
    records_with_errors = 0
    records_with_warnings = 0
    total_actions = 0
    
    # =========================================================================
    # 数据集特有统计（从 checker 返回的 stats 中累加数值类型字段）
    # =========================================================================
    dataset_specific_stats = {}  # 累加的数值统计
    
    # 详细结果
    record_results = []
    error_records = []
    
    for record in data_iterator:
        if max_samples and total_records >= max_samples:
            break
        
        total_records += 1
        
        # 检查可执行性 - checker 返回 (errors, warnings, stats)
        errors, warnings, stats = static_checker.check(record)
        
        # Record 级别统计
        if errors:
            records_with_errors += 1
            error_records.append({
                'sample_id': record.sample_id,
                'annotation_id': record.metadata.get('annotation_id', ''),
                'errors': errors,
            })
        if warnings:
            records_with_warnings += 1
        
        # 通用 Action 统计
        total_actions += stats.get('total_actions', 0)
        
        # 累加数据集特有的数值统计（跳过 rate 类型的字段，这些需要最后重新计算）
        for key, value in stats.items():
            if isinstance(value, (int, float)) and key != 'total_actions':
                # 跳过比率字段（包含 rate 的字段不应该累加）
                if 'rate' in key.lower():
                    continue
                if key not in dataset_specific_stats:
                    dataset_specific_stats[key] = 0
                dataset_specific_stats[key] += value
        
        # 记录详细结果（保留 checker 返回的所有 stats，包括 action_results）
        record_result = {
            'sample_id': record.sample_id,
            'annotation_id': record.metadata.get('annotation_id', ''),
            'website': record.website,
            'errors': errors,
            'warnings': warnings,
        }
        # 合并 stats 中的所有字段（包括 action_results）
        for key, value in stats.items():
            record_result[key] = value
        record_results.append(record_result)
        
        # 进度
        if progress_interval and total_records % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = total_records / elapsed if elapsed > 0 else 0
            print(f"  [{total_records:,}] {rate:.2f} 条/秒")
    
    elapsed = time.time() - start_time
    
    # =========================================================================
    # 构建结果
    # =========================================================================
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        
        # 通用统计
        'total_records': total_records,
        'records_with_errors': records_with_errors,
        'records_with_warnings': records_with_warnings,
        'total_actions': total_actions,
        
        # 详细结果
        'record_results': record_results,
        'error_records': error_records,
    }
    
    # 合并数据集特有统计，并计算比率
    # WebShop: 
    #   - success_rate = success_count / total_actions (动作执行成功率)
    #   - task_completion_rate = task_completed / total_records (任务完成率)
    #   - task_success_rate = task_success / total_records (任务成功率，完成且reward>0)
    if 'success_count' in dataset_specific_stats and total_actions > 0:
        dataset_specific_stats['action_success_rate'] = dataset_specific_stats['success_count'] / total_actions
    if 'task_completed' in dataset_specific_stats and total_records > 0:
        dataset_specific_stats['task_completion_rate'] = dataset_specific_stats['task_completed'] / total_records
    if 'task_success' in dataset_specific_stats and total_records > 0:
        dataset_specific_stats['task_success_rate'] = dataset_specific_stats['task_success'] / total_records
    if 'task_partial' in dataset_specific_stats and total_records > 0:
        dataset_specific_stats['task_partial_rate'] = dataset_specific_stats['task_partial'] / total_records
    # final_reward 改为平均值
    if 'final_reward' in dataset_specific_stats and total_records > 0:
        dataset_specific_stats['avg_reward'] = dataset_specific_stats['final_reward'] / total_records
        del dataset_specific_stats['final_reward']  # 删除累加值，只保留平均值
    
    # Mind2Web: coord_rate, attr_rate
    verified = dataset_specific_stats.get('verified_actions', 0)
    if verified > 0:
        if 'coord_success' in dataset_specific_stats:
            dataset_specific_stats['coord_rate'] = dataset_specific_stats['coord_success'] / verified
        if 'attr_success' in dataset_specific_stats:
            dataset_specific_stats['attr_rate'] = dataset_specific_stats['attr_success'] / verified
    
    results['dataset_specific_stats'] = dataset_specific_stats
    
    # 保存结果
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
        
        # 生成简洁的汇总 log 文件
        summary_file = output_file.replace('.json', '_summary.txt')
        _save_summary_log(results, summary_file, elapsed, dataset_name)
    
    # 打印摘要
    _print_summary(results, elapsed, dataset_name)
    
    return results


def _save_summary_log(results: Dict[str, Any], summary_file: str, elapsed: float, dataset_name: str):
    """保存简洁的汇总 log 文件"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Static Executability 评估汇总 - {dataset_name}")
    lines.append("=" * 60)
    lines.append(f"时间: {results.get('timestamp', 'N/A')}")
    lines.append(f"耗时: {elapsed:.1f} 秒")
    lines.append("")
    
    lines.append("【Record 级别】")
    lines.append(f"  总 Record 数: {results['total_records']:,}")
    lines.append(f"  有错误: {results['records_with_errors']:,}")
    lines.append(f"  有警告: {results['records_with_warnings']:,}")
    lines.append("")
    
    lines.append("【Action 级别】")
    lines.append(f"  总 Action 数: {results['total_actions']:,}")
    lines.append("")
    
    # 数据集特有统计
    dataset_stats = results.get('dataset_specific_stats', {})
    if dataset_stats:
        lines.append(f"【{dataset_name} 特有指标】")
        for key, value in sorted(dataset_stats.items()):
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value:,}")
        lines.append("")
    
    # 关键指标汇总（放在最显眼的位置）
    lines.append("=" * 60)
    lines.append("【关键指标汇总】")
    lines.append("=" * 60)
    
    if 'task_success_rate' in dataset_stats:
        lines.append(f"  ✅ 任务成功率 (reward=1.0): {dataset_stats['task_success_rate']:.2%}")
    if 'task_partial_rate' in dataset_stats:
        lines.append(f"  ⚠️ 部分成功率 (0<reward<1): {dataset_stats['task_partial_rate']:.2%}")
    if 'avg_reward' in dataset_stats:
        lines.append(f"  📊 平均 reward: {dataset_stats['avg_reward']:.4f}")
    if 'coord_rate' in dataset_stats:
        lines.append(f"  📍 坐标定位成功率: {dataset_stats['coord_rate']:.2%}")
    if 'attr_rate' in dataset_stats:
        lines.append(f"  🏷️ 属性定位成功率: {dataset_stats['attr_rate']:.2%}")
    
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"详细结果: {summary_file.replace('_summary.txt', '.json')}")
    lines.append("=" * 60)
    
    # 写入文件
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"汇总已保存到: {summary_file}")


def _print_summary(results: Dict[str, Any], elapsed: float, dataset_name: str):
    """打印评估摘要"""
    print()
    print("=" * 70)
    print(f"评估完成！耗时 {elapsed:.1f} 秒")
    print("=" * 70)
    print()
    print(f"【Record 级别】")
    print(f"  总 Record 数: {results['total_records']:,}")
    print(f"  有错误: {results['records_with_errors']:,}")
    print(f"  有警告: {results['records_with_warnings']:,}")
    print()
    print(f"【Action 级别】")
    print(f"  总 Action 数: {results['total_actions']:,}")
    print()
    
    # 打印数据集特有统计
    dataset_stats = results.get('dataset_specific_stats', {})
    if dataset_stats:
        print(f"【{dataset_name} 特有指标】")
        for key, value in sorted(dataset_stats.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:,}")
    print()
    
    # 按网站统计（如果有多个网站）
    website_stats = {}
    for r in results['record_results']:
        site = r.get('website') or 'unknown'
        if site not in website_stats:
            website_stats[site] = {'records': 0, 'actions': 0}
        website_stats[site]['records'] += 1
        website_stats[site]['actions'] += r.get('total_actions', 0)
    
    if len(website_stats) > 1:
        print(f"【按网站统计】")
        sorted_sites = sorted(website_stats.items(), key=lambda x: -x[1]['records'])
        for site, stats in sorted_sites[:10]:
            print(f"  {site}: {stats['records']} records, {stats['actions']} actions")
        if len(sorted_sites) > 10:
            print(f"  ... 还有 {len(sorted_sites) - 10} 个网站")
        print()


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="静态可执行性指标评估")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["mind2web", "webshop", "weblinx"],
                        help="数据集名称")
    parser.add_argument("--data-path", type=str, default=None,
                        help="数据集路径")
    parser.add_argument("--raw-dump", type=str, default=None,
                        help="raw_dump 路径 (Mind2Web)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数（用于测试）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--show", action="store_true",
                        help="显示浏览器窗口")
    parser.add_argument("--progress-interval", type=int, default=10,
                        help="进度显示间隔")
    
    args = parser.parse_args()
    
    # 根据数据集选择 loader 和 checker
    if args.dataset == "mind2web":
        from loaders import Mind2WebLoader
        from mind2web_executor import Mind2WebStaticChecker
        
        data_path = args.data_path or '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/Mind2Web/data'
        loader = Mind2WebLoader(data_path)
        loader.load()
        
        checker = Mind2WebStaticChecker(
            raw_dump_path=args.raw_dump,
            headless=not args.show,
        )
        dataset_name = "Mind2Web"
        
    elif args.dataset == "webshop":
        from loaders import WebShopLoader
        from webshop_executor import WebShopStaticChecker
        
        data_path = args.data_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'webshop/baseline_models/data/il_trajs_finalized_images.jsonl')
        loader = WebShopLoader(data_path)
        
        checker = WebShopStaticChecker(
            use_browser=args.show,  # --show 表示使用 browser 模式
            render=args.show,
        )
        dataset_name = "WebShop"
        
    elif args.dataset == "weblinx":
        # TODO: WebLINX checker 还未实现
        raise NotImplementedError("WebLINX static checker not implemented yet")
    
    # 设置输出文件
    output_file = args.output
    if output_file is None:
        output_file = f"results/{args.dataset}/static_executability_results.json"
    
    # 运行评估
    results = compute_static_executability(
        data_iterator=loader.iterate(),
        static_checker=checker,
        dataset_name=dataset_name,
        output_file=output_file,
        max_samples=args.max_samples,
        progress_interval=args.progress_interval,
    )
