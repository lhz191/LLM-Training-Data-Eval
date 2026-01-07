#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Format Check 指标 - 数据格式验证

检查数据集样本的格式正确性，包括：
- 必需字段是否存在
- 字段格式是否正确
- 标签是否配对等

使用方式:
    from format_check import compute_format_check
    from loaders import OpenMathInstructLoader
    from openmath_executor import OpenMathFormatChecker
    
    loader = OpenMathInstructLoader('/path/to/OpenMathInstruct-1')
    checker = OpenMathFormatChecker()
    
    results = compute_format_check(
        data_iterator=loader.iterate(),
        format_checker=checker,
        dataset_name='OpenMathInstruct-1',
        output_file='format_check_results.json'
    )
"""

import json
import time
from datetime import datetime
from typing import Optional, Iterator, Dict, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from data_types import MathSample
from code_executor import FormatChecker


# =============================================================================
# 并行处理辅助函数
# =============================================================================

def _check_single_sample(sample: MathSample, checker_class: type) -> Dict[str, Any]:
    """
    检查单个样本的格式（用于并行处理）
    
    Args:
        sample: MathSample 样本
        checker_class: FormatChecker 类（不是实例，因为需要在子进程中创建）
    
    Returns:
        检查结果字典
    """
    checker = checker_class()
    errors, warnings = checker.check(sample)
    
    return {
        'sample_id': sample.sample_id,
        'errors': errors,
        'warnings': warnings,
        'has_errors': len(errors) > 0,
        'has_warnings': len(warnings) > 0,
    }


def compute_format_check(
    data_iterator: Iterator[MathSample],
    format_checker: FormatChecker,
    dataset_name: str = "unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 10000,
) -> Dict[str, Any]:
    """
    计算 Format Check 指标
    
    Args:
        data_iterator: MathSample 迭代器
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
    
    # 统计
    total = 0
    passed = 0  # 无错误
    with_errors = 0  # 有错误
    with_warnings = 0  # 有警告（但无错误）
    
    # 详细结果
    error_samples = []  # 有错误的样本
    warning_samples = []  # 只有警告的样本
    
    # 当前批次的错误样本（用于进度输出）
    batch_error_ids = []
    
    for sample in data_iterator:
        if max_samples and total >= max_samples:
            break
        
        total += 1
        
        # 检查格式
        errors, warnings = format_checker.check(sample)
        
        if errors:
            with_errors += 1
            batch_error_ids.append(sample.sample_id)
            error_samples.append({
                'sample_id': sample.sample_id,
                'errors': errors,
                'warnings': warnings,
            })
        elif warnings:
            with_warnings += 1
            passed += 1  # 只有警告也算通过
            warning_samples.append({
                'sample_id': sample.sample_id,
                'warnings': warnings,
            })
        else:
            passed += 1
        
        # 进度
        if progress_interval and total % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            pass_rate = passed / total if total > 0 else 0
            print(f"  [{total}/{max_samples or '?'}] {rate:.1f} 条/秒, 通过率: {pass_rate:.2%}")
            # 显示当前批次的错误样本
            if batch_error_ids:
                # 排序后输出
                try:
                    sorted_ids = sorted(batch_error_ids, key=lambda x: int(x.split('_')[-1]))
                except:
                    sorted_ids = sorted(batch_error_ids)
                print(f"    格式错误样本: {sorted_ids}")
                batch_error_ids = []
    
    elapsed = time.time() - start_time
    
    # 计算比率
    pass_rate = passed / total if total > 0 else 0
    error_rate = with_errors / total if total > 0 else 0
    warning_rate = with_warnings / total if total > 0 else 0
    
    # 结果
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        
        # 统计
        'total': total,
        'passed': passed,
        'with_errors': with_errors,
        'with_warnings': with_warnings,
        
        # 比率
        'pass_rate': pass_rate,
        'error_rate': error_rate,
        'warning_rate': warning_rate,
        
        # 详细样本（完整保存）
        'error_samples': error_samples,
        'warning_samples': warning_samples,
    }
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
    
    # 打印摘要
    print()
    print("=" * 70)
    print(f"评估完成！耗时 {elapsed:.1f} 秒")
    print("=" * 70)
    print()
    print(f"总样本数: {total:,}")
    print(f"通过: {passed:,} ({pass_rate:.2%})")
    print(f"格式错误: {with_errors:,} ({error_rate:.2%})")
    print(f"有警告: {with_warnings:,} ({warning_rate:.2%})")
    print()
    
    # 显示错误类型统计
    if error_samples:
        print("【错误类型统计】")
        error_types = {}
        for sample in error_samples:
            for err in sample['errors']:
                # 提取错误类型（去掉具体数字）
                err_type = err.split(':')[0] if ':' in err else err
                error_types[err_type] = error_types.get(err_type, 0) + 1
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {err_type}: {count:,}")
        print()
    
    # 显示警告类型统计
    if warning_samples:
        print("【警告类型统计】")
        warning_types = {}
        for sample in warning_samples:
            for warn in sample['warnings']:
                warn_type = warn.split(':')[0] if ':' in warn else warn
                warning_types[warn_type] = warning_types.get(warn_type, 0) + 1
        for warn_type, count in sorted(warning_types.items(), key=lambda x: -x[1]):
            print(f"  {warn_type}: {count:,}")
        print()
    
    return results


# =============================================================================
# 并行版本
# =============================================================================

def compute_format_check_parallel(
    data_iterator: Iterator[MathSample],
    format_checker_class: type,
    dataset_name: str = "unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 10000,
    max_workers: Optional[int] = None,
    batch_size: int = 10000,
) -> Dict[str, Any]:
    """
    计算 Format Check 指标（并行版本）
    
    Args:
        data_iterator: MathSample 迭代器
        format_checker_class: FormatChecker 类（不是实例）
        dataset_name: 数据集名称
        output_file: 结果输出文件
        max_samples: 最大样本数（用于测试）
        progress_interval: 进度显示间隔
        max_workers: 最大并行数（默认为 CPU 核心数）
        batch_size: 每批处理的样本数
    
    Returns:
        结果字典
    """
    if max_workers is None:
        max_workers = min(32, cpu_count())  # 限制最大线程数
    
    print("=" * 70)
    print("Format Check Evaluation (并行模式)")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"并行线程: {max_workers}")
    print(f"批大小: {batch_size:,}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(flush=True)
    
    start_time = time.time()
    
    # 统计
    completed = 0
    passed = 0
    with_errors = 0
    with_warnings = 0
    
    # 详细结果
    error_samples = []
    warning_samples = []
    
    # 当前批次的错误样本
    batch_error_ids = []
    
    # 分批处理，避免一次性加载所有样本
    def process_batch(batch_samples):
        nonlocal completed, passed, with_errors, with_warnings, batch_error_ids
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_check_single_sample, sample, format_checker_class): sample.sample_id
                for sample in batch_samples
            }
            
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                
                if result['has_errors']:
                    with_errors += 1
                    batch_error_ids.append(result['sample_id'])
                    error_samples.append({
                        'sample_id': result['sample_id'],
                        'errors': result['errors'],
                        'warnings': result['warnings'],
                    })
                elif result['has_warnings']:
                    with_warnings += 1
                    passed += 1
                    warning_samples.append({
                        'sample_id': result['sample_id'],
                        'warnings': result['warnings'],
                    })
                else:
                    passed += 1
                
                # 进度
                if progress_interval and completed % progress_interval == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    pass_rate = passed / completed if completed > 0 else 0
                    total_str = f"{max_samples:,}" if max_samples else "?"
                    print(f"  [{completed:,}/{total_str}] {rate:.1f} 条/秒, 通过率: {pass_rate:.2%}", flush=True)
                    # 显示当前批次的错误样本（排序后输出）
                    if batch_error_ids:
                        try:
                            sorted_ids = sorted(batch_error_ids, key=lambda x: int(x.split('_')[-1]))
                        except:
                            sorted_ids = sorted(batch_error_ids)
                        print(f"    格式错误样本: {sorted_ids}", flush=True)
                        batch_error_ids = []
    
    # 分批读取和处理
    current_batch = []
    total_loaded = 0
    
    print("开始处理...", flush=True)
    
    for sample in data_iterator:
        if max_samples and total_loaded >= max_samples:
            break
        
        current_batch.append(sample)
        total_loaded += 1
        
        # 当批次满时处理
        if len(current_batch) >= batch_size:
            process_batch(current_batch)
            current_batch = []
    
    # 处理剩余的样本
    if current_batch:
        process_batch(current_batch)
    
    elapsed = time.time() - start_time
    total = completed  # 实际处理的样本数
    
    # 计算比率
    pass_rate = passed / total if total > 0 else 0
    error_rate = with_errors / total if total > 0 else 0
    warning_rate = with_warnings / total if total > 0 else 0
    
    # 结果
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'max_workers': max_workers,
        
        # 统计
        'total': total,
        'passed': passed,
        'with_errors': with_errors,
        'with_warnings': with_warnings,
        
        # 比率
        'pass_rate': pass_rate,
        'error_rate': error_rate,
        'warning_rate': warning_rate,
        
        # 详细样本（完整保存）
        'error_samples': error_samples,
        'warning_samples': warning_samples,
    }
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
    
    # 打印摘要
    print()
    print("=" * 70)
    print(f"评估完成！耗时 {elapsed:.1f} 秒")
    print("=" * 70)
    print()
    print(f"总样本数: {total:,}")
    print(f"通过: {passed:,} ({pass_rate:.2%})")
    print(f"格式错误: {with_errors:,} ({error_rate:.2%})")
    print(f"有警告: {with_warnings:,} ({warning_rate:.2%})")
    print()
    
    # 显示错误类型统计
    if error_samples:
        print("【错误类型统计】")
        error_types = {}
        for sample in error_samples:
            for err in sample['errors']:
                err_type = err.split(':')[0] if ':' in err else err
                error_types[err_type] = error_types.get(err_type, 0) + 1
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {err_type}: {count:,}")
        print()
    
    # 显示警告类型统计
    if warning_samples:
        print("【警告类型统计】")
        warning_types = {}
        for sample in warning_samples:
            for warn in sample['warnings']:
                warn_type = warn.split(':')[0] if ':' in warn else warn
                warning_types[warn_type] = warning_types.get(warn_type, 0) + 1
        for warn_type, count in sorted(warning_types.items(), key=lambda x: -x[1]):
            print(f"  {warn_type}: {count:,}")
        print()
    
    return results


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Format Check 指标评估")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["openmathinstruct", "lila"],
                        help="数据集名称")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数（用于测试）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--parallel", action="store_true",
                        help="使用并行模式")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行线程数（默认为 CPU 核心数）")
    
    args = parser.parse_args()
    
    # 根据数据集选择 loader 和 checker
    if args.dataset == "openmathinstruct":
        from loaders import OpenMathInstructLoader
        from openmath_executor import OpenMathFormatChecker
        
        loader = OpenMathInstructLoader(
            '/mnt/petrelfs/liuhaoze/datasets/Symbolic_and_Logical_Data/OpenMathInstruct-1',
            use_correct=True
        )
        checker_class = OpenMathFormatChecker
        dataset_name = "OpenMathInstruct-1"
        
    elif args.dataset == "lila":
        from loaders import LILALoader
        from lila_executor import LILAFormatChecker
        
        loader = LILALoader(
            '/mnt/petrelfs/liuhaoze/datasets/Symbolic_and_Logical_Data/LILA/lila/multi/iid/train_math_only.json'
        )
        checker_class = LILAFormatChecker
        dataset_name = "LILA"
    
    # 设置输出文件
    output_file = args.output
    if output_file is None:
        output_file = f"results/{args.dataset}/format_check_results.json"
    
    # 运行评估
    if args.parallel:
        results = compute_format_check_parallel(
            data_iterator=loader.iterate(),
            format_checker_class=checker_class,
            dataset_name=dataset_name,
            output_file=output_file,
            max_samples=args.max_samples,
            progress_interval=10000,
            max_workers=args.workers,
        )
    else:
        checker = checker_class()
        results = compute_format_check(
            data_iterator=loader.iterate(),
            format_checker=checker,
            dataset_name=dataset_name,
            output_file=output_file,
            max_samples=args.max_samples,
            progress_interval=10000,
        )

