#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML 信息保留率指标

计算 raw_html 和 cleaned_html 的定位能力，评估清洗过程中关键信息的保留程度。

指标定义：
- raw_localization_rate: 在 raw_html 中能定位到 target 的比例
- clean_localization_rate: 在 cleaned_html 中能定位到 target 的比例
- retention_rate: 信息保留率 = clean_success / raw_success

用途：
- 评估数据预处理/清洗对定位能力的影响
- 发现 clean_html 中丢失关键信息的问题
- 比较不同数据集的信息保留质量

使用方式：
    from metrics.html_retention import compute_html_retention
    from text_gui_executor import get_html_locator
    from loaders import Mind2WebLoader
    
    loader = Mind2WebLoader('/path/to/data')
    records = loader.parse_all()
    
    locator = get_html_locator('mind2web')
    results = compute_html_retention(records, locator, output_file='results.json')
"""

import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from data_types import Record
from text_gui_executor import HTMLLocator


def compute_html_retention(
    records: List[Record],
    locator: HTMLLocator,
    output_file: Optional[str] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    计算 HTML 信息保留率
    
    Args:
        records: Record 列表
        locator: HTML 定位器实例
        output_file: 输出 JSON 文件路径（可选）
        show_progress: 是否显示进度
        
    Returns:
        结果字典，包含：
        - total_actions: 总 action 数
        - raw_success: raw_html 定位成功数
        - raw_failed: raw_html 定位失败数
        - raw_rate: raw_html 定位率
        - clean_success: clean_html 定位成功数
        - clean_failed: clean_html 定位失败数
        - clean_rate: clean_html 定位率
        - retention_rate: 信息保留率
        - details: 每条记录的详细结果
    """
    start_time = time.time()
    
    # 统计
    total_actions = 0
    raw_success = 0
    raw_failed = 0
    clean_success = 0
    clean_failed = 0
    
    # 失败原因统计
    raw_fail_reasons = {}
    clean_fail_reasons = {}
    
    # 详细结果
    details = []
    
    for i, record in enumerate(records):
        if show_progress and (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1:,}] {rate:.1f} 条/秒")
        
        record_result = {
            'sample_id': record.sample_id,
            'total_actions': len(record.actions),
            'raw_success': 0,
            'clean_success': 0,
            'actions': []
        }
        
        for j, action in enumerate(record.actions):
            total_actions += 1
            
            # 检查 raw_html 定位
            raw_html = action.raw_html or ''
            raw_ok, raw_reason = locator.can_locate(action, raw_html)
            
            # 检查 cleaned_html 定位
            clean_html = action.cleaned_html or ''
            clean_ok, clean_reason = locator.can_locate(action, clean_html)
            
            # 统计
            if raw_ok:
                raw_success += 1
                record_result['raw_success'] += 1
            else:
                raw_failed += 1
                raw_fail_reasons[raw_reason] = raw_fail_reasons.get(raw_reason, 0) + 1
            
            if clean_ok:
                clean_success += 1
                record_result['clean_success'] += 1
            else:
                clean_failed += 1
                clean_fail_reasons[clean_reason] = clean_fail_reasons.get(clean_reason, 0) + 1
            
            # 记录动作结果
            action_result = {
                'action_idx': j,
                'action_type': action.action_type,
                'raw_html_len': len(raw_html),
                'clean_html_len': len(clean_html),
                'raw_can_locate': raw_ok,
                'raw_reason': raw_reason,
                'clean_can_locate': clean_ok,
                'clean_reason': clean_reason,
            }
            record_result['actions'].append(action_result)
        
        details.append(record_result)
    
    elapsed = time.time() - start_time
    
    # 计算比率
    raw_rate = raw_success / total_actions if total_actions > 0 else 0
    clean_rate = clean_success / total_actions if total_actions > 0 else 0
    retention_rate = clean_success / raw_success if raw_success > 0 else 0
    
    # 汇总结果
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(records),
            'total_actions': total_actions,
            'elapsed_seconds': round(elapsed, 2),
        },
        'summary': {
            'total_actions': total_actions,
            'raw_success': raw_success,
            'raw_failed': raw_failed,
            'raw_rate': round(raw_rate, 4),
            'clean_success': clean_success,
            'clean_failed': clean_failed,
            'clean_rate': round(clean_rate, 4),
            'retention_rate': round(retention_rate, 4),
        },
        'fail_reasons': {
            'raw': raw_fail_reasons,
            'clean': clean_fail_reasons,
        },
        'details': details,
    }
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")
        
        # 保存汇总
        summary_file = output_file.replace('.json', '_summary.txt')
        _save_summary(results, summary_file)
        print(f"汇总已保存到: {summary_file}")
    
    # 打印汇总
    _print_summary(results)
    
    return results


def _save_summary(results: Dict, output_file: str):
    """保存汇总到文本文件"""
    summary = results['summary']
    fail_reasons = results['fail_reasons']
    
    lines = [
        "=" * 70,
        "HTML 信息保留率 - 汇总报告",
        "=" * 70,
        f"生成时间: {results['metadata']['timestamp']}",
        f"总记录数: {results['metadata']['total_records']:,}",
        f"总 Action 数: {summary['total_actions']:,}",
        f"耗时: {results['metadata']['elapsed_seconds']:.2f} 秒",
        "",
        "-" * 70,
        "定位统计",
        "-" * 70,
        "",
        "【Raw HTML 定位】",
        f"  成功: {summary['raw_success']:,} ({summary['raw_rate']*100:.2f}%)",
        f"  失败: {summary['raw_failed']:,}",
        "",
        "【Clean HTML 定位】",
        f"  成功: {summary['clean_success']:,} ({summary['clean_rate']*100:.2f}%)",
        f"  失败: {summary['clean_failed']:,}",
        "",
        "【信息保留率】",
        f"  retention_rate = clean_success / raw_success",
        f"                 = {summary['clean_success']:,} / {summary['raw_success']:,}",
        f"                 = {summary['retention_rate']*100:.2f}%",
        "",
        "-" * 70,
        "失败原因统计",
        "-" * 70,
        "",
        "【Raw HTML 失败原因】",
    ]
    
    for reason, count in sorted(fail_reasons['raw'].items(), key=lambda x: -x[1]):
        lines.append(f"  {reason}: {count:,}")
    
    if not fail_reasons['raw']:
        lines.append("  (无)")
    
    lines.extend([
        "",
        "【Clean HTML 失败原因】",
    ])
    
    for reason, count in sorted(fail_reasons['clean'].items(), key=lambda x: -x[1]):
        lines.append(f"  {reason}: {count:,}")
    
    if not fail_reasons['clean']:
        lines.append("  (无)")
    
    lines.append("")
    lines.append("=" * 70)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _print_summary(results: Dict):
    """打印汇总信息"""
    summary = results['summary']
    
    print()
    print("=" * 70)
    print("HTML 信息保留率")
    print("=" * 70)
    print()
    print(f"【Raw HTML 定位】")
    print(f"  成功: {summary['raw_success']:,} / {summary['total_actions']:,} ({summary['raw_rate']*100:.2f}%)")
    print()
    print(f"【Clean HTML 定位】")
    print(f"  成功: {summary['clean_success']:,} / {summary['total_actions']:,} ({summary['clean_rate']*100:.2f}%)")
    print()
    print(f"【信息保留率】")
    print(f"  {summary['retention_rate']*100:.2f}% (clean_success / raw_success)")
    print()
