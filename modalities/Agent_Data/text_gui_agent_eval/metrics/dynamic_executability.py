#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Executability 指标 - GUI Agent 动态可执行性检查

在真实网站上执行 GUI Agent 数据集样本的操作：
- 坐标定位成功率：使用 bounding_box 坐标能否定位到目标元素
- 属性定位成功率：使用 tag/class/id 等属性能否定位到目标元素
- 执行成功率：操作能否成功执行

使用方式:
    from metrics import compute_dynamic_executability
    from loaders import Mind2WebLoader
    from mind2web_executor import Mind2WebDynamicChecker
    
    loader = Mind2WebLoader('/path/to/mind2web/data')
    checker = Mind2WebDynamicChecker(headless=False)
    
    results = compute_dynamic_executability(
        data_iterator=loader.iterate(),
        dynamic_checker=checker,
        dataset_name='Mind2Web',
        output_file='dynamic_executability_results.json'
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
from text_gui_executor import DynamicExecutabilityChecker


def compute_dynamic_executability(
    data_iterator: Iterator[Record],
    dynamic_checker: DynamicExecutabilityChecker,
    dataset_name: str = "unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 1,
    execute: bool = True,
) -> Dict[str, Any]:
    """
    计算动态可执行性指标
    
    Args:
        data_iterator: Record 迭代器
        dynamic_checker: 动态可执行性检查器
        dataset_name: 数据集名称
        output_file: 结果输出文件
        max_samples: 最大样本数（用于测试）
        progress_interval: 进度显示间隔
        execute: 是否执行操作（默认 True）
    
    Returns:
        结果字典，包含：
        - coord_rate: 坐标定位成功率
        - attr_rate: 属性定位成功率
        - exec_rate: 执行成功率
        - 详细的 record 和 action 级别结果
    """
    print("=" * 70)
    print("Dynamic Executability Evaluation")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"执行模式: {'执行操作' if execute else '仅验证'}")
    print()
    
    start_time = time.time()
    
    # Record 级别统计
    total_records = 0
    records_with_errors = 0
    records_with_warnings = 0
    
    # Action 级别统计
    total_actions = 0
    coord_success = 0
    attr_success = 0
    exec_success = 0
    
    # 详细结果
    record_results = []
    error_records = []
    
    for record in data_iterator:
        if max_samples and total_records >= max_samples:
            break
        
        total_records += 1
        
        # 检查动态可执行性
        errors, warnings, stats = dynamic_checker.check(
            record, 
            execute=execute,
        )
        
        # Record 级别统计
        if errors:
            records_with_errors += 1
            error_records.append({
                'sample_id': record.sample_id,
                'website': record.website,
                'errors': errors,
            })
        if warnings:
            records_with_warnings += 1
        
        # Action 级别统计
        total_actions += stats.get('total_actions', 0)
        coord_success += stats.get('coords_success', 0)
        attr_success += stats.get('attrs_success', 0)
        exec_success += stats.get('executed_actions', 0)
        
        # 记录详细结果
        record_results.append({
            'sample_id': record.sample_id,
            'website': record.website,
            'url': stats.get('url', ''),
            'total_actions': stats.get('total_actions', 0),
            'coords_success': stats.get('coords_success', 0),
            'attrs_success': stats.get('attrs_success', 0),
            'executed_actions': stats.get('executed_actions', 0),
            'coord_rate': stats.get('coords_rate', 0.0),
            'attr_rate': stats.get('attrs_rate', 0.0),
            'exec_rate': stats.get('execution_rate', 0.0),
            'errors': errors,
            'warnings': warnings,
            'action_results': stats.get('action_results', []),
        })
        
        # 进度
        if progress_interval and total_records % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = total_records / elapsed if elapsed > 0 else 0
            curr_coord_rate = coord_success / total_actions if total_actions > 0 else 0
            curr_attr_rate = attr_success / total_actions if total_actions > 0 else 0
            curr_exec_rate = exec_success / total_actions if total_actions > 0 else 0
            print(f"\n  [{total_records:,}] {rate:.2f} 条/秒")
            print(f"    坐标定位: {coord_success}/{total_actions} ({curr_coord_rate:.1%})")
            print(f"    属性定位: {attr_success}/{total_actions} ({curr_attr_rate:.1%})")
            if execute:
                print(f"    执行成功: {exec_success}/{total_actions} ({curr_exec_rate:.1%})")
    
    elapsed = time.time() - start_time
    
    # 计算总体成功率
    coord_rate = coord_success / total_actions if total_actions > 0 else 0.0
    attr_rate = attr_success / total_actions if total_actions > 0 else 0.0
    exec_rate = exec_success / total_actions if total_actions > 0 else 0.0
    
    # 结果
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'execute_mode': execute,
        
        # Record 级别统计
        'total_records': total_records,
        'records_with_errors': records_with_errors,
        'records_with_warnings': records_with_warnings,
        
        # Action 级别统计
        'total_actions': total_actions,
        'coord_success': coord_success,
        'attr_success': attr_success,
        'exec_success': exec_success,
        
        # 成功率
        'coord_rate': coord_rate,
        'attr_rate': attr_rate,
        'exec_rate': exec_rate,
        
        # 详细结果
        'record_results': record_results,
        'error_records': error_records,
    }
    
    # 保存结果
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
    
        # 保存 summary.txt
        summary_file = output_file.replace('.json', '_summary.txt')
        _save_summary_log(results, summary_file, elapsed, output_file, execute)
    
    # 打印摘要
    _print_summary(results, elapsed, execute)
    
    return results


def _save_summary_log(results: Dict[str, Any], summary_file: str, elapsed: float, json_file: str, execute: bool):
    """保存简洁的 summary.txt 日志"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Dynamic Executability 评估汇总 - {results.get('dataset', 'Unknown')}")
    lines.append("=" * 60)
    lines.append(f"时间: {results.get('timestamp', 'N/A')}")
    lines.append(f"耗时: {elapsed:.1f} 秒")
    lines.append(f"执行模式: {'执行操作' if execute else '仅验证'}")
    lines.append("")
    
    lines.append("【Record 级别】")
    lines.append(f"  总 Record 数: {results.get('total_records', 0)}")
    lines.append(f"  有错误: {results.get('records_with_errors', 0)}")
    lines.append(f"  有警告: {results.get('records_with_warnings', 0)}")
    lines.append("")
    
    lines.append("【Action 级别】")
    lines.append(f"  总 Action 数: {results.get('total_actions', 0)}")
    lines.append("")
    
    lines.append("【动态可执行性指标】")
    lines.append(f"  coord_rate: {results.get('coord_rate', 0):.4f}")
    lines.append(f"  coord_success: {results.get('coord_success', 0)}")
    lines.append(f"  attr_rate: {results.get('attr_rate', 0):.4f}")
    lines.append(f"  attr_success: {results.get('attr_success', 0)}")
    if execute:
        lines.append(f"  exec_rate: {results.get('exec_rate', 0):.4f}")
        lines.append(f"  exec_success: {results.get('exec_success', 0)}")
    lines.append("")
    
    lines.append("=" * 60)
    lines.append("【关键指标汇总】")
    lines.append("=" * 60)
    coord_rate = results.get('coord_rate', 0)
    attr_rate = results.get('attr_rate', 0)
    exec_rate = results.get('exec_rate', 0)
    lines.append(f"  📍 坐标定位成功率: {coord_rate:.2%}")
    lines.append(f"  🏷️ 属性定位成功率: {attr_rate:.2%}")
    if execute:
        lines.append(f"  ⚡ 执行成功率: {exec_rate:.2%}")
    lines.append("")
    
    # 按网站统计
    website_stats = {}
    for r in results.get('record_results', []):
        site = r.get('website') or 'unknown'
        if site not in website_stats:
            website_stats[site] = {
                'records': 0,
                'actions': 0,
                'coord': 0,
                'attr': 0,
                'exec': 0,
            }
        website_stats[site]['records'] += 1
        website_stats[site]['actions'] += r.get('total_actions', 0)
        website_stats[site]['coord'] += r.get('coords_success', 0)
        website_stats[site]['attr'] += r.get('attrs_success', 0)
        website_stats[site]['exec'] += r.get('executed_actions', 0)
    
    if len(website_stats) > 1:
        lines.append("【按网站统计 (Top 10)】")
        sorted_sites = sorted(website_stats.items(), key=lambda x: -x[1]['records'])
        for site, stats in sorted_sites[:10]:
            a = stats['actions']
            if a > 0:
                coord_r = stats['coord'] / a
                attr_r = stats['attr'] / a
                exec_r = stats['exec'] / a
                if execute:
                    lines.append(f"  {site}: {stats['records']} records, "
                                f"坐标 {coord_r:.0%}, 属性 {attr_r:.0%}, 执行 {exec_r:.0%}")
                else:
                    lines.append(f"  {site}: {stats['records']} records, "
                                f"坐标 {coord_r:.0%}, 属性 {attr_r:.0%}")
        if len(sorted_sites) > 10:
            lines.append(f"  ... 还有 {len(sorted_sites) - 10} 个网站")
        lines.append("")
    
    lines.append("=" * 60)
    lines.append(f"详细结果: {json_file}")
    lines.append("=" * 60)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"汇总已保存到: {summary_file}")


def _print_summary(results: Dict[str, Any], elapsed: float, execute: bool):
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
    print(f"【动态可执行性】")
    print(f"  坐标定位成功率: {results['coord_success']:,}/{results['total_actions']:,} "
          f"({results['coord_rate']:.2%})")
    print(f"  属性定位成功率: {results['attr_success']:,}/{results['total_actions']:,} "
          f"({results['attr_rate']:.2%})")
    if execute:
        print(f"  执行成功率: {results['exec_success']:,}/{results['total_actions']:,} "
              f"({results['exec_rate']:.2%})")
    print()
    
    # 按网站统计
    website_stats = {}
    for r in results['record_results']:
        site = r.get('website') or 'unknown'
        if site not in website_stats:
            website_stats[site] = {
                'records': 0,
                'actions': 0,
                'coord': 0,
                'attr': 0,
                'exec': 0,
            }
        website_stats[site]['records'] += 1
        website_stats[site]['actions'] += r.get('total_actions', 0)
        website_stats[site]['coord'] += r.get('coords_success', 0)
        website_stats[site]['attr'] += r.get('attrs_success', 0)
        website_stats[site]['exec'] += r.get('executed_actions', 0)
    
    if len(website_stats) > 1:
        print(f"【按网站统计】")
        sorted_sites = sorted(website_stats.items(), key=lambda x: -x[1]['records'])
        for site, stats in sorted_sites[:10]:
            a = stats['actions']
            if a > 0:
                coord_r = stats['coord'] / a
                attr_r = stats['attr'] / a
                exec_r = stats['exec'] / a
                if execute:
                    print(f"  {site}: {stats['records']} records, "
                          f"坐标 {coord_r:.0%}, 属性 {attr_r:.0%}, 执行 {exec_r:.0%}")
                else:
                    print(f"  {site}: {stats['records']} records, "
                          f"坐标 {coord_r:.0%}, 属性 {attr_r:.0%}")
        if len(sorted_sites) > 10:
            print(f"  ... 还有 {len(sorted_sites) - 10} 个网站")
        print()


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="动态可执行性指标评估")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["mind2web", "weblinx"],
                        help="数据集名称")
    parser.add_argument("--data-path", type=str, default=None,
                        help="数据集路径")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数（用于测试）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--show", action="store_true",
                        help="显示浏览器窗口")
    parser.add_argument("--no-execute", action="store_true",
                        help="仅验证不执行操作")
    parser.add_argument("--progress-interval", type=int, default=1,
                        help="进度显示间隔")
    
    args = parser.parse_args()
    
    # 根据数据集选择 loader 和 checker
    if args.dataset == "mind2web":
        from loaders import Mind2WebLoader
        from mind2web_executor import Mind2WebDynamicChecker
        
        data_path = args.data_path or '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/Mind2Web/data'
        loader = Mind2WebLoader(data_path)
        loader.load()
        
        checker = Mind2WebDynamicChecker(
            headless=not args.show,
        )
        dataset_name = "Mind2Web"
        
    elif args.dataset == "weblinx":
        # TODO: WebLINX checker 还未实现
        raise NotImplementedError("WebLINX dynamic checker not implemented yet")
    
    # 设置输出文件
    output_file = args.output
    if output_file is None:
        output_file = f"results/{args.dataset}/dynamic_executability_results.json"
    
    # 运行评估
    results = compute_dynamic_executability(
        data_iterator=loader.iterate(),
        dynamic_checker=checker,
        dataset_name=dataset_name,
        output_file=output_file,
        max_samples=args.max_samples,
        progress_interval=args.progress_interval,
        execute=not args.no_execute,
    )
