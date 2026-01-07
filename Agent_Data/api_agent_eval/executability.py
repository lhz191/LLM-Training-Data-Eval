#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Executability 指标 - 静态可执行性检查

检查 API Agent 数据集样本的可执行性，包括：
- API 是否在工具列表中定义
- 必需参数是否完整
- 参数类型是否匹配
- final_answer 是否可从 API 响应推导（ToolBench）

使用方式:
    from executability import compute_executability
    from loaders import ToolBenchLoader
    from api_executor import get_executability_checker
    
    loader = ToolBenchLoader('/path/to/toolbench.json')
    checker = get_executability_checker('toolbench', toolenv_path='/path/to/toolenv')
    
    results = compute_executability(
        data_iterator=loader.iterate(),
        executability_checker=checker,
        dataset_name='ToolBench',
        output_file='executability_results.json'
    )
"""

import json
import time
from datetime import datetime
from typing import Optional, Iterator, Dict, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from data_types import APIAgentSample
from api_executor import ExecutabilityChecker


# =============================================================================
# 并行处理辅助函数
# =============================================================================

def _check_single_sample_executability(args) -> Dict[str, Any]:
    """
    检查单个样本的可执行性（用于并行处理）
    
    Args:
        args: (sample, checker_class, checker_kwargs) 元组
    
    Returns:
        检查结果字典
    """
    sample, checker_class, checker_kwargs = args
    checker = checker_class(**checker_kwargs)
    errors, warnings, stats = checker.check(sample)
    
    return {
        'sample_id': sample.sample_id,
        'errors': errors,
        'warnings': warnings,
        'stats': stats,
        'has_errors': len(errors) > 0,
        'has_warnings': len(warnings) > 0,
    }


def compute_executability(
    data_iterator: Iterator[APIAgentSample],
    executability_checker: ExecutabilityChecker,
    dataset_name: str = "unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 10000,
) -> Dict[str, Any]:
    """
    计算 Executability 指标（串行版本）
    
    Args:
        data_iterator: APIAgentSample 迭代器
        executability_checker: 可执行性检查器
        dataset_name: 数据集名称
        output_file: 结果输出文件
        max_samples: 最大样本数（用于测试）
        progress_interval: 进度显示间隔
    
    Returns:
        结果字典
    """
    print("=" * 70)
    print("Executability Evaluation")
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
    
    # API 调用统计
    total_api_calls = 0
    
    # 可推导性统计（ToolBench）
    derivability_total = 0
    derivability_passed = 0
    
    # 详细结果
    error_samples = []
    warning_samples = []
    
    # 当前批次的错误样本
    batch_error_ids = []
    
    for sample in data_iterator:
        if max_samples and total >= max_samples:
            break
        
        total += 1
        
        # 检查可执行性
        errors, warnings, stats = executability_checker.check(sample)
        
        # 统计 API 调用
        total_api_calls += stats.get('api_calls_checked', 0)
        
        # 统计可推导性
        if stats.get('train_derivability'):
            derivability_total += 1
            if stats['train_derivability'].get('derivable'):
                derivability_passed += 1
        
        if errors:
            with_errors += 1
            batch_error_ids.append(sample.sample_id)
            error_samples.append({
                'sample_id': sample.sample_id,
                'errors': errors,
                'warnings': warnings,
                'stats': stats,
            })
        elif warnings:
            with_warnings += 1
            passed += 1
            warning_samples.append({
                'sample_id': sample.sample_id,
                'warnings': warnings,
                'stats': stats,
            })
        else:
            passed += 1
        
        # 进度
        if progress_interval and total % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            pass_rate = passed / total if total > 0 else 0
            print(f"  [{total:,}/{max_samples or '?'}] {rate:.1f} 条/秒, 通过率: {pass_rate:.2%}")
            if batch_error_ids:
                try:
                    sorted_ids = sorted(batch_error_ids, key=lambda x: int(str(x).split('_')[-1]))
                except:
                    sorted_ids = sorted(batch_error_ids, key=str)
                print(f"    可执行性错误: {sorted_ids[:10]}{'...' if len(sorted_ids) > 10 else ''}")
                batch_error_ids = []
    
    elapsed = time.time() - start_time
    
    # 计算比率
    pass_rate = passed / total if total > 0 else 0
    error_rate = with_errors / total if total > 0 else 0
    warning_rate = with_warnings / total if total > 0 else 0
    derivability_rate = derivability_passed / derivability_total if derivability_total > 0 else 0
    
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
        
        # API 调用统计
        'total_api_calls': total_api_calls,
        'avg_api_calls_per_sample': total_api_calls / total if total > 0 else 0,
        
        # 可推导性统计
        'derivability': {
            'total': derivability_total,
            'passed': derivability_passed,
            'rate': derivability_rate,
        },
        
        # 详细样本（完整保存）
        'error_samples': error_samples,
        'warning_samples': warning_samples,
    }
    
    # 保存结果
    if output_file:
        import os
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
    
    # 打印摘要
    _print_summary(results, elapsed, error_samples, derivability_total, derivability_passed, derivability_rate, total_api_calls, total)
    
    return results


def compute_executability_parallel(
    data_iterator: Iterator[APIAgentSample],
    executability_checker_class: type,
    checker_kwargs: Dict[str, Any] = None,
    dataset_name: str = "unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 10000,
    max_workers: Optional[int] = None,
    batch_size: int = 10000,
) -> Dict[str, Any]:
    """
    计算 Executability 指标（并行版本）
    
    Args:
        data_iterator: APIAgentSample 迭代器
        executability_checker_class: ExecutabilityChecker 类（不是实例）
        checker_kwargs: 传递给 checker 的参数
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
        max_workers = min(32, cpu_count())
    
    if checker_kwargs is None:
        checker_kwargs = {}
    
    print("=" * 70)
    print("Executability Evaluation (并行模式)")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"并行进程: {max_workers}")
    print(f"批大小: {batch_size:,}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(flush=True)
    
    start_time = time.time()
    
    # 统计
    completed = 0
    passed = 0
    with_errors = 0
    with_warnings = 0
    total_api_calls = 0
    derivability_total = 0
    derivability_passed = 0
    
    # 详细结果
    error_samples = []
    warning_samples = []
    
    # 当前批次的错误样本
    batch_error_ids = []
    
    # 分批处理
    def process_batch(batch_samples):
        nonlocal completed, passed, with_errors, with_warnings, batch_error_ids
        nonlocal total_api_calls, derivability_total, derivability_passed
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _check_single_sample_executability, 
                    (sample, executability_checker_class, checker_kwargs)
                ): sample.sample_id
                for sample in batch_samples
            }
            
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                
                # 统计 API 调用
                stats = result.get('stats', {})
                total_api_calls += stats.get('api_calls_checked', 0)
                
                # 统计可推导性
                if stats.get('train_derivability'):
                    derivability_total += 1
                    if stats['train_derivability'].get('derivable'):
                        derivability_passed += 1
                
                if result['has_errors']:
                    with_errors += 1
                    batch_error_ids.append(result['sample_id'])
                    error_samples.append({
                        'sample_id': result['sample_id'],
                        'errors': result['errors'],
                        'warnings': result['warnings'],
                        'stats': result['stats'],
                    })
                elif result['has_warnings']:
                    with_warnings += 1
                    passed += 1
                    warning_samples.append({
                        'sample_id': result['sample_id'],
                        'warnings': result['warnings'],
                        'stats': result['stats'],
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
                    if batch_error_ids:
                        try:
                            sorted_ids = sorted(batch_error_ids, key=lambda x: int(str(x).split('_')[-1]))
                        except:
                            sorted_ids = sorted(batch_error_ids, key=str)
                        print(f"    可执行性错误: {sorted_ids[:10]}{'...' if len(sorted_ids) > 10 else ''}", flush=True)
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
        
        if len(current_batch) >= batch_size:
            process_batch(current_batch)
            current_batch = []
    
    # 处理剩余的样本
    if current_batch:
        process_batch(current_batch)
    
    elapsed = time.time() - start_time
    total = completed
    
    # 计算比率
    pass_rate = passed / total if total > 0 else 0
    error_rate = with_errors / total if total > 0 else 0
    warning_rate = with_warnings / total if total > 0 else 0
    derivability_rate = derivability_passed / derivability_total if derivability_total > 0 else 0
    
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
        
        # API 调用统计
        'total_api_calls': total_api_calls,
        'avg_api_calls_per_sample': total_api_calls / total if total > 0 else 0,
        
        # 可推导性统计
        'derivability': {
            'total': derivability_total,
            'passed': derivability_passed,
            'rate': derivability_rate,
        },
        
        # 详细样本（完整保存）
        'error_samples': error_samples,
        'warning_samples': warning_samples,
    }
    
    # 保存结果
    if output_file:
        import os
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
    
    # 打印摘要
    _print_summary(results, elapsed, error_samples, derivability_total, derivability_passed, derivability_rate, total_api_calls, total)
    
    return results


def _print_summary(results, elapsed, error_samples, derivability_total, derivability_passed, derivability_rate, total_api_calls, total):
    """打印评估摘要"""
    print()
    print("=" * 70)
    print(f"评估完成！耗时 {elapsed:.1f} 秒")
    print("=" * 70)
    print()
    print(f"总样本数: {results['total']:,}")
    print(f"通过: {results['passed']:,} ({results['pass_rate']:.2%})")
    print(f"可执行性错误: {results['with_errors']:,} ({results['error_rate']:.2%})")
    print(f"有警告: {results['with_warnings']:,} ({results['warning_rate']:.2%})")
    print()
    print(f"【API 调用统计】")
    print(f"  总 API 调用数: {total_api_calls:,}")
    print(f"  平均每样本 API 调用: {total_api_calls / total if total > 0 else 0:.2f}")
    print()
    
    if derivability_total > 0:
        print(f"【可推导性统计】")
        print(f"  检查样本数: {derivability_total:,}")
        print(f"  可推导: {derivability_passed:,} ({derivability_rate:.2%})")
        print()
    
    # 显示错误类型统计
    if error_samples:
        print("【错误类型统计】")
        error_types = {}
        for sample in error_samples:
            for err in sample['errors']:
                err_type = err.split(':')[0] if ':' in err else err
                error_types[err_type] = error_types.get(err_type, 0) + 1
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1])[:20]:
            print(f"  {err_type}: {count:,}")
        print()


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Executability 指标评估")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["toolbench", "xlam"],
                        help="数据集名称")
    parser.add_argument("--toolenv-path", type=str, default=None,
                        help="ToolBench toolenv 路径")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数（用于测试）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--parallel", action="store_true",
                        help="使用并行模式")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行进程数（默认为 CPU 核心数）")
    parser.add_argument("--progress-interval", type=int, default=10000,
                        help="进度显示间隔")
    
    args = parser.parse_args()
    
    # 根据数据集选择 loader 和 checker
    if args.dataset == "toolbench":
        from loaders import ToolBenchLoader
        from toolbench_executor import ToolBenchExecutabilityChecker
        
        loader = ToolBenchLoader(
            '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/toolbench_official/toolllama_G123_dfs_train.json'
        )
        
        toolenv_path = args.toolenv_path or '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/toolbench_official/toolenv/tools'
        checker_class = ToolBenchExecutabilityChecker
        checker_kwargs = {'toolenv_path': toolenv_path}
        dataset_name = "ToolBench"
        
    elif args.dataset == "xlam":
        from loaders import XLAMLoader
        from xlam_executor import XLAMExecutabilityChecker
        
        loader = XLAMLoader(
            '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/xlam_60k.jsonl'
        )
        checker_class = XLAMExecutabilityChecker
        checker_kwargs = {}
        dataset_name = "xLAM-60k"
    
    # 设置输出文件
    output_file = args.output
    if output_file is None:
        output_file = f"results/{args.dataset}/executability_results.json"
    
    # 运行评估
    if args.parallel:
        results = compute_executability_parallel(
            data_iterator=loader.iterate(),
            executability_checker_class=checker_class,
            checker_kwargs=checker_kwargs,
            dataset_name=dataset_name,
            output_file=output_file,
            max_samples=args.max_samples,
            progress_interval=args.progress_interval,
            max_workers=args.workers,
        )
    else:
        checker = checker_class(**checker_kwargs)
        results = compute_executability(
            data_iterator=loader.iterate(),
            executability_checker=checker,
            dataset_name=dataset_name,
            output_file=output_file,
            max_samples=args.max_samples,
            progress_interval=args.progress_interval,
        )
