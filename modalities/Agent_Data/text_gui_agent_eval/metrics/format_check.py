#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Format Check 指标 - GUI Agent 数据格式验证

检查 Text GUI Agent 数据集样本的格式正确性，包括：
- 必需字段是否存在
- 字段格式是否正确
- 数据一致性检查（如 target 在 candidates 中）

使用方式:
    from metrics.format_check import compute_format_check
    from loaders import Mind2WebLoader
    from mind2web_executor import Mind2WebFormatChecker
    
    loader = Mind2WebLoader('/path/to/mind2web/data')
    checker = Mind2WebFormatChecker()
    
    results = compute_format_check(
        data_iterator=loader.iterate(),
        format_checker=checker,
        dataset_name='Mind2Web',
        output_file='format_check_results.json'
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
from text_gui_executor import FormatChecker


def compute_format_check(
    data_iterator: Iterator[Record],
    format_checker: FormatChecker,
    dataset_name: str = "unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 100,
) -> Dict[str, Any]:
    """
    计算 Format Check 指标
    
    Args:
        data_iterator: Record 迭代器
        format_checker: 格式检查器
        dataset_name: 数据集名称
        output_file: 结果输出文件
        max_samples: 最大样本数（用于测试）
        progress_interval: 进度显示间隔
    
    Returns:
        结果字典
    """
    print("=" * 70)
    print("Format Check Evaluation")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    # Record 级别统计
    total_records = 0
    passed_records = 0  # 无错误
    records_with_errors = 0
    records_with_warnings = 0
    
    # Action 级别统计
    total_actions = 0
    actions_with_errors = 0
    total_error_count = 0  # 总错误数（一个 action 可能有多个错误）
    
    # 详细结果
    record_results = []
    error_records = []
    warning_records = []
    
    for record in data_iterator:
        if max_samples and total_records >= max_samples:
            break
        
        total_records += 1
        
        # 检查格式
        errors, warnings = format_checker.check(record)
        
        # 统计 action 数
        num_actions = len(record.actions)
        total_actions += num_actions
        
        # 统计有错误的 action 数（通过解析 Action[x] 前缀）
        error_action_indices = set()
        for err in errors:
            if err.startswith('Action['):
                try:
                    idx = int(err.split(']')[0].replace('Action[', ''))
                    error_action_indices.add(idx)
                except:
                    pass
        actions_with_errors += len(error_action_indices)
        total_error_count += len(errors)
        
        # 记录结果
        result = {
            'sample_id': record.sample_id,
            'website': record.website,
            'total_actions': num_actions,
            'actions_with_errors': len(error_action_indices),
            'error_count': len(errors),
            'errors': errors,
            'warnings': warnings,
            'has_errors': len(errors) > 0,
            'has_warnings': len(warnings) > 0,
        }
        record_results.append(result)
        
        if errors:
            records_with_errors += 1
            error_records.append(result)
        elif warnings:
            records_with_warnings += 1
            passed_records += 1  # 只有警告也算通过
            warning_records.append(result)
        else:
            passed_records += 1
        
        # 进度
        if progress_interval and total_records % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = total_records / elapsed if elapsed > 0 else 0
            pass_rate = passed_records / total_records if total_records > 0 else 0
            print(f"  [{total_records:,}] {rate:.1f} 条/秒, 通过率: {pass_rate:.2%}")
    
    elapsed = time.time() - start_time
    
    # 计算比率
    pass_rate = passed_records / total_records if total_records > 0 else 0
    error_rate = records_with_errors / total_records if total_records > 0 else 0
    warning_rate = records_with_warnings / total_records if total_records > 0 else 0
    
    # Action 级别比率
    passed_actions = total_actions - actions_with_errors
    action_pass_rate = passed_actions / total_actions if total_actions > 0 else 0
    action_error_rate = actions_with_errors / total_actions if total_actions > 0 else 0
    
    # 结果
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        
        # Record 级别统计
        'total_records': total_records,
        'passed_records': passed_records,
        'records_with_errors': records_with_errors,
        'records_with_warnings': records_with_warnings,
        
        # Action 级别统计
        'total_actions': total_actions,
        'passed_actions': passed_actions,
        'actions_with_errors': actions_with_errors,
        'total_error_count': total_error_count,
        
        # Record 比率
        'pass_rate': pass_rate,
        'error_rate': error_rate,
        'warning_rate': warning_rate,
        
        # Action 比率
        'action_pass_rate': action_pass_rate,
        'action_error_rate': action_error_rate,
        
        # 详细结果
        'record_results': record_results,
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
        _save_summary_log(results, summary_file, elapsed, output_file)
    
    # 打印摘要
    _print_summary(results, elapsed)
    
    return results


def _save_summary_log(results: Dict[str, Any], summary_file: str, elapsed: float, json_file: str):
    """保存简洁的 summary.txt 日志"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Format Check 评估汇总 - {results.get('dataset', 'Unknown')}")
    lines.append("=" * 60)
    lines.append(f"时间: {results.get('timestamp', 'N/A')}")
    lines.append(f"耗时: {elapsed:.1f} 秒")
    lines.append("")
    
    total_records = results.get('total_records', 0)
    passed_records = results.get('passed_records', 0)
    records_with_errors = results.get('records_with_errors', 0)
    total_actions = results.get('total_actions', 0)
    passed_actions = results.get('passed_actions', 0)
    actions_with_errors = results.get('actions_with_errors', 0)
    total_error_count = results.get('total_error_count', 0)
    pass_rate = results.get('pass_rate', 0)
    action_pass_rate = results.get('action_pass_rate', 0)
    
    lines.append("【Record 级别】")
    lines.append(f"  总 Record 数: {total_records:,}")
    lines.append(f"  通过: {passed_records:,} ({pass_rate:.2%})")
    lines.append(f"  有错误: {records_with_errors:,} ({results.get('error_rate', 0):.2%})")
    lines.append("")
    
    lines.append("【Action 级别】")
    lines.append(f"  总 Action 数: {total_actions:,}")
    lines.append(f"  通过: {passed_actions:,} ({action_pass_rate:.2%})")
    lines.append(f"  有错误: {actions_with_errors:,} ({results.get('action_error_rate', 0):.2%})")
    lines.append(f"  总错误数: {total_error_count:,}")
    lines.append("")
    
    lines.append("=" * 60)
    lines.append("【关键指标汇总】")
    lines.append("=" * 60)
    lines.append(f"  ✅ Record 通过率: {pass_rate:.2%}")
    lines.append(f"  ✅ Action 通过率: {action_pass_rate:.2%}")
    lines.append("")
    
    # 错误类型统计 - 提取 "Action[x]: xxx" 中的 xxx 部分
    error_types = {}
    for r in results.get('record_results', []):
        for err in r.get('errors', []):
            # 格式: "Action[x]: error message" -> 取 "error message"
            if ': ' in err:
                err_type = err.split(': ', 1)[1]
            else:
                err_type = err
            error_types[err_type] = error_types.get(err_type, 0) + 1
    
    if error_types:
        lines.append("【错误类型统计 (Top 10)】")
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {err_type}: {count}")
        lines.append("")
    
    # 警告类型统计 - 提取 "Action[x]: xxx" 中的 xxx 部分
    warning_types = {}
    for r in results.get('record_results', []):
        for warn in r.get('warnings', []):
            if ': ' in warn:
                warn_type = warn.split(': ', 1)[1]
            else:
                warn_type = warn
            warning_types[warn_type] = warning_types.get(warn_type, 0) + 1
    
    if warning_types:
        lines.append("【警告类型统计 (Top 10)】")
        for warn_type, count in sorted(warning_types.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {warn_type}: {count}")
        lines.append("")
    
    lines.append("=" * 60)
    lines.append(f"详细结果: {json_file}")
    lines.append("=" * 60)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"汇总已保存到: {summary_file}")


def _print_summary(results: Dict[str, Any], elapsed: float):
    """打印评估摘要"""
    print()
    print("=" * 70)
    print(f"评估完成！耗时 {elapsed:.1f} 秒")
    print("=" * 70)
    print()
    print(f"【Record 级别】")
    print(f"  总 Record 数: {results['total_records']:,}")
    print(f"  通过: {results['passed_records']:,} ({results['pass_rate']:.2%})")
    print(f"  有错误: {results['records_with_errors']:,} ({results['error_rate']:.2%})")
    print()
    print(f"【Action 级别】")
    print(f"  总 Action 数: {results['total_actions']:,}")
    print(f"  通过: {results['passed_actions']:,} ({results['action_pass_rate']:.2%})")
    print(f"  有错误: {results['actions_with_errors']:,} ({results['action_error_rate']:.2%})")
    print(f"  总错误数: {results['total_error_count']:,}")
    print()
    
    # 显示错误类型统计 - 提取 "Action[x]: xxx" 中的 xxx 部分
    error_types = {}
    for r in results.get('record_results', []):
        for err in r.get('errors', []):
            if ': ' in err:
                err_type = err.split(': ', 1)[1]
            else:
                err_type = err
            error_types[err_type] = error_types.get(err_type, 0) + 1
    
    if error_types:
        print("【错误类型统计】")
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1])[:20]:
            print(f"  {err_type}: {count:,}")
        print()
    
    # 显示警告类型统计 - 提取 "Action[x]: xxx" 中的 xxx 部分
    warning_types = {}
    for r in results.get('record_results', []):
        for warn in r.get('warnings', []):
            if ': ' in warn:
                warn_type = warn.split(': ', 1)[1]
            else:
                warn_type = warn
            warning_types[warn_type] = warning_types.get(warn_type, 0) + 1
    
    if warning_types:
        print("【警告类型统计】")
        for warn_type, count in sorted(warning_types.items(), key=lambda x: -x[1])[:20]:
            print(f"  {warn_type}: {count:,}")
        print()


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Format Check 指标评估")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["mind2web", "webshop"],
                        help="数据集名称")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数（用于测试）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--progress-interval", type=int, default=100,
                        help="进度显示间隔")
    
    args = parser.parse_args()
    
    # 根据数据集选择 loader 和 checker
    if args.dataset == "mind2web":
        from loaders import Mind2WebLoader
        from mind2web_executor import Mind2WebFormatChecker
        
        loader = Mind2WebLoader('/home/liuhaoze/Desktop/mind2web')
        checker = Mind2WebFormatChecker()
        dataset_name = "Mind2Web"
        
    elif args.dataset == "webshop":
        from loaders import WebShopLoader
        from webshop_executor import WebShopFormatChecker
        
        webshop_data = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'webshop/baseline_models/data/il_trajs_finalized_images.jsonl'
        )
        loader = WebShopLoader(webshop_data)
        checker = WebShopFormatChecker()
        dataset_name = "WebShop"
    
    # 设置输出文件
    output_file = args.output
    if output_file is None:
        output_file = f"results/{args.dataset}/format_check_results.json"
    
    # 运行评估
    results = compute_format_check(
        data_iterator=loader.iterate(),
        format_checker=checker,
        dataset_name=dataset_name,
        output_file=output_file,
        max_samples=args.max_samples,
        progress_interval=args.progress_interval,
    )
