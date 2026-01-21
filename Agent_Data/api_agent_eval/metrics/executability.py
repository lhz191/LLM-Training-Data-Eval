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

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_types import APIAgentSample
from api_executor import ExecutabilityChecker


# =============================================================================
# 辅助函数
# =============================================================================

def _compute_error_type_distribution(error_samples: List[Dict], total: int) -> Dict[str, Any]:
    """
    计算错误类型分布统计
    
    将错误分为两大类：
    1. API 错误：toolenv 找不到 API、必需参数缺失等
    2. LLM Judge 错误：Train Derivability、Query-Answer Relevance
    
    Args:
        error_samples: 错误样本列表
        total: 总样本数
    
    Returns:
        错误类型分布统计字典
    """
    api_only_errors = 0  # 只有 API 相关错误
    llm_only_errors = 0  # 只有 LLM Judge 错误
    both_errors = 0      # 两者都有
    api_error_ids = set()
    llm_error_ids = set()
    
    for sample in error_samples:
        sample_id = sample['sample_id']
        errors = sample['errors']
        
        has_api_error = False
        has_llm_error = False
        
        for err in errors:
            if 'Train Derivability' in err or 'Query-Answer Relevance' in err:
                has_llm_error = True
            else:
                has_api_error = True
        
        if has_api_error:
            api_error_ids.add(sample_id)
        if has_llm_error:
            llm_error_ids.add(sample_id)
        
        if has_api_error and has_llm_error:
            both_errors += 1
        elif has_api_error:
            api_only_errors += 1
        elif has_llm_error:
            llm_only_errors += 1
    
    # 纯静态可执行性（只看 API 错误）
    api_error_count = len(api_error_ids)
    pure_static_passed = total - api_error_count
    
    return {
        'api_only_errors': api_only_errors,
        'llm_only_errors': llm_only_errors,
        'both_errors': both_errors,
        'api_error_samples': api_error_count,
        'llm_error_samples': len(llm_error_ids),
        'pure_static_executability': {
            'passed': pure_static_passed,
            'failed': api_error_count,
            'pass_rate': pure_static_passed / total if total > 0 else 0,
        }
    }


# =============================================================================
# 并行处理辅助函数
# =============================================================================

# 全局变量，用于在 worker 进程中缓存 checker 实例
_worker_checker = None
_worker_checker_class = None
_worker_checker_kwargs = None


def _init_worker(checker_class, checker_kwargs):
    """
    Worker 进程初始化函数
    在每个 worker 启动时只调用一次，创建并缓存 checker 实例
    """
    global _worker_checker, _worker_checker_class, _worker_checker_kwargs
    _worker_checker_class = checker_class
    _worker_checker_kwargs = checker_kwargs
    _worker_checker = checker_class(**checker_kwargs)
    # 预先构建 API 映射（如果是 ToolBench）
    if hasattr(_worker_checker, '_build_api_mapping'):
        _worker_checker._build_api_mapping()


def _check_single_sample_executability(sample) -> Dict[str, Any]:
    """
    检查单个样本的可执行性（用于并行处理）
    
    使用 worker 进程中缓存的 checker 实例，避免重复初始化
    
    Args:
        sample: APIAgentSample 对象
    
    Returns:
        检查结果字典
    """
    global _worker_checker
    
    # 如果 checker 还没初始化（理论上不应该发生）
    if _worker_checker is None:
        _worker_checker = _worker_checker_class(**_worker_checker_kwargs)
        if hasattr(_worker_checker, '_build_api_mapping'):
            _worker_checker._build_api_mapping()
    
    errors, warnings, stats = _worker_checker.check(sample)
    
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
    
    # 可推导性统计（LLM Judge）
    derivability_total = 0
    derivability_passed = 0
    
    # 查询相关性统计（LLM Judge）
    relevance_total = 0
    relevance_passed = 0
    
    # 详细结果
    error_samples = []
    warning_samples = []
    
    # LLM Judge 详细结果
    llm_judge_results = []
    
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
        
        # 统计可推导性（Train Derivability）
        if stats.get('train_derivability'):
            derivability_total += 1
            if stats['train_derivability'].get('derivable'):
                derivability_passed += 1
            # 记录 LLM Judge 结果
            llm_judge_results.append({
                'sample_id': sample.sample_id,
                'type': 'train_derivability',
                'result': stats['train_derivability'].get('derivable'),
                'reason': stats['train_derivability'].get('reason', ''),
            })
        
        # 统计查询相关性（Query-Answer Relevance）
        if stats.get('query_relevance'):
            relevance_total += 1
            if stats['query_relevance'].get('relevant'):
                relevance_passed += 1
            # 记录 LLM Judge 结果
            llm_judge_results.append({
                'sample_id': sample.sample_id,
                'type': 'query_relevance',
                'result': stats['query_relevance'].get('relevant'),
                'reason': stats['query_relevance'].get('reason', ''),
            })
        
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
                print(f"    可执行性错误: {sorted_ids}")
                batch_error_ids = []
    
    elapsed = time.time() - start_time
    
    # 计算比率
    pass_rate = passed / total if total > 0 else 0
    error_rate = with_errors / total if total > 0 else 0
    warning_rate = with_warnings / total if total > 0 else 0
    derivability_rate = derivability_passed / derivability_total if derivability_total > 0 else 0
    relevance_rate = relevance_passed / relevance_total if relevance_total > 0 else 0
    
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
        
        # LLM Judge 统计 - 可推导性（Train Derivability）
        'derivability': {
            'total': derivability_total,
            'passed': derivability_passed,
            'rate': derivability_rate,
        },
        
        # LLM Judge 统计 - 查询相关性（Query-Answer Relevance）
        'relevance': {
            'total': relevance_total,
            'passed': relevance_passed,
            'rate': relevance_rate,
        },
        
        # LLM Judge 详细结果（包含每个样本的判断理由）
        'llm_judge_results': llm_judge_results,
        
        # 错误类型分布统计
        'error_type_distribution': _compute_error_type_distribution(error_samples, total),
        
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
    _print_summary(results, elapsed, error_samples, derivability_total, derivability_passed, derivability_rate, 
                   relevance_total, relevance_passed, relevance_rate, total_api_calls, total)
    
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
    relevance_total = 0
    relevance_passed = 0
    
    # 详细结果
    error_samples = []
    warning_samples = []
    
    # LLM Judge 详细结果
    llm_judge_results = []
    
    # 当前批次的错误样本
    batch_error_ids = []
    
    # 先收集所有样本（避免迭代器与进程池冲突）
    print("Step 1: 收集样本...", flush=True)
    all_samples = []
    for sample in data_iterator:
        if max_samples and len(all_samples) >= max_samples:
            break
        all_samples.append(sample)
    
    total_to_process = len(all_samples)
    print(f"共 {total_to_process:,} 个样本待处理", flush=True)
    print()
    
    # Step 2: 创建一次进程池，处理所有样本
    print("Step 2: 并行检查...", flush=True)
    
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(executability_checker_class, checker_kwargs)
    ) as executor:
        # 提交所有任务
        futures = {
            executor.submit(_check_single_sample_executability, sample): sample.sample_id
            for sample in all_samples
        }
        
        # 处理结果
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            
            # 统计 API 调用
            stats = result.get('stats', {})
            total_api_calls += stats.get('api_calls_checked', 0)
            
            # 统计可推导性（Train Derivability）
            if stats.get('train_derivability'):
                derivability_total += 1
                if stats['train_derivability'].get('derivable'):
                    derivability_passed += 1
                # 记录 LLM Judge 结果
                llm_judge_results.append({
                    'sample_id': result['sample_id'],
                    'type': 'train_derivability',
                    'result': stats['train_derivability'].get('derivable'),
                    'reason': stats['train_derivability'].get('reason', ''),
                })
            
            # 统计查询相关性（Query-Answer Relevance）
            if stats.get('query_relevance'):
                relevance_total += 1
                if stats['query_relevance'].get('relevant'):
                    relevance_passed += 1
                # 记录 LLM Judge 结果
                llm_judge_results.append({
                    'sample_id': result['sample_id'],
                    'type': 'query_relevance',
                    'result': stats['query_relevance'].get('relevant'),
                    'reason': stats['query_relevance'].get('reason', ''),
                })
            
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
                print(f"  [{completed:,}/{total_to_process:,}] {rate:.1f} 条/秒, 通过率: {pass_rate:.2%}", flush=True)
                if batch_error_ids:
                    try:
                        sorted_ids = sorted(batch_error_ids, key=lambda x: int(str(x).split('_')[-1]))
                    except:
                        sorted_ids = sorted(batch_error_ids, key=str)
                    print(f"    可执行性错误: {sorted_ids}", flush=True)
                    batch_error_ids = []
    
    elapsed = time.time() - start_time
    total = completed
    
    # 计算比率
    pass_rate = passed / total if total > 0 else 0
    error_rate = with_errors / total if total > 0 else 0
    warning_rate = with_warnings / total if total > 0 else 0
    derivability_rate = derivability_passed / derivability_total if derivability_total > 0 else 0
    relevance_rate = relevance_passed / relevance_total if relevance_total > 0 else 0
    
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
        
        # LLM Judge 统计 - 可推导性（Train Derivability）
        'derivability': {
            'total': derivability_total,
            'passed': derivability_passed,
            'rate': derivability_rate,
        },
        
        # LLM Judge 统计 - 查询相关性（Query-Answer Relevance）
        'relevance': {
            'total': relevance_total,
            'passed': relevance_passed,
            'rate': relevance_rate,
        },
        
        # LLM Judge 详细结果（包含每个样本的判断理由）
        'llm_judge_results': llm_judge_results,
        
        # 错误类型分布统计
        'error_type_distribution': _compute_error_type_distribution(error_samples, total),
        
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
    _print_summary(results, elapsed, error_samples, derivability_total, derivability_passed, derivability_rate, 
                   relevance_total, relevance_passed, relevance_rate, total_api_calls, total)
    
    return results


def _print_summary(results, elapsed, error_samples, derivability_total, derivability_passed, derivability_rate,
                   relevance_total, relevance_passed, relevance_rate, total_api_calls, total):
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
    
    # LLM Judge 统计
    if derivability_total > 0 or relevance_total > 0:
        print(f"【LLM Judge 统计】")
        if derivability_total > 0:
            print(f"  可推导性 (Train Derivability):")
            print(f"    检查样本数: {derivability_total:,}")
            print(f"    通过: {derivability_passed:,} ({derivability_rate:.2%})")
        if relevance_total > 0:
            print(f"  查询相关性 (Query-Answer Relevance):")
            print(f"    检查样本数: {relevance_total:,}")
            print(f"    通过: {relevance_passed:,} ({relevance_rate:.2%})")
            print()
    
    # 显示错误类型分布
    if 'error_type_distribution' in results:
        dist = results['error_type_distribution']
        print("【错误类型分布】")
        print(f"  只有 API 错误（toolenv/参数）: {dist['api_only_errors']:,}")
        print(f"  只有 LLM Judge 错误: {dist['llm_only_errors']:,}")
        print(f"  两者都有: {dist['both_errors']:,}")
        print()
        print(f"  纯静态可执行性（不含 LLM Judge）:")
        pure = dist['pure_static_executability']
        print(f"    通过: {pure['passed']:,} ({pure['pass_rate']:.2%})")
        print(f"    失败: {pure['failed']:,} ({1 - pure['pass_rate']:.2%})")
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
