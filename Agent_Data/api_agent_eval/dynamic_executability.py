#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Executability 指标 - 动态可执行性检查

通过实际调用 RapidAPI 验证 API 调用的有效性：
- API 是否真实存在且可访问
- 参数格式是否正确
- 是否能获得有效响应
- final_answer 是否能从真实 API 返回推导

注意：此检查需要 RapidAPI Key 和网络访问权限。

使用方式:
    from dynamic_executability import compute_dynamic_executability
    from loaders import ToolBenchLoader
    from toolbench_executor import ToolBenchDynamicChecker
    
    loader = ToolBenchLoader('/path/to/toolbench.json')
    checker = ToolBenchDynamicChecker(
        rapidapi_key='your_key',
        toolenv_path='/path/to/toolenv'
    )
    
    results = compute_dynamic_executability(
        data_iterator=loader.iterate(),
        dynamic_checker=checker,
        dataset_name='ToolBench',
        output_file='dynamic_executability_results.json'
    )
"""

import json
import time
import random
from datetime import datetime
from typing import Optional, Iterator, Dict, List, Any
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_types import APIAgentSample


def _check_single_sample_dynamic(args) -> Dict[str, Any]:
    """
    检查单个样本的动态可执行性（用于并行处理）
    
    Args:
        args: (sample, checker) 元组
    
    Returns:
        检查结果字典
    """
    sample, checker = args
    return checker.check_sample(sample)


def compute_dynamic_executability(
    data_iterator: Iterator[APIAgentSample],
    dynamic_checker,  # ToolBenchDynamicChecker
    dataset_name: str = "unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 100,
    max_workers: int = 16,
) -> Dict[str, Any]:
    """
    计算 Dynamic Executability 指标（支持多线程并行）
    
    Args:
        data_iterator: APIAgentSample 迭代器
        dynamic_checker: 动态检查器（如 ToolBenchDynamicChecker）
        dataset_name: 数据集名称
        output_file: 结果输出文件
        max_samples: 最大样本数（用于测试，None 表示全量）
        progress_interval: 进度显示间隔
        max_workers: 并行线程数（网络 I/O 密集型，可以设大一些）
    
    Returns:
        结果字典
    """
    print("=" * 70)
    print("Dynamic Executability Evaluation")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"并行线程: {max_workers}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 收集所有样本
    print("收集样本...")
    all_samples = []
    for sample in data_iterator:
        all_samples.append(sample)
        if max_samples and len(all_samples) >= max_samples:
            break
    
    total_samples = len(all_samples)
    print(f"待检查样本数: {total_samples:,}")
    print()
    
    # 构建 API 映射（在主线程做一次）
    if hasattr(dynamic_checker, '_build_api_mapping') and dynamic_checker.toolenv_path:
        print("构建 API 映射...")
        dynamic_checker._build_api_mapping()
    
    start_time = time.time()
    
    # 统计
    results_data = {
        'total_samples': total_samples,
        'passed_samples': 0,
        'failed_samples': 0,
        'api_call_stats': {
            'total': 0,
            'success': 0,
            'failed': 0,
            'not_found': 0,
        },
        'finish_stats': {
            'give_answer': 0,
            'give_up_and_restart': 0,
            'invalid': 0,
        },
        'derivability': {
            'total_checked': 0,
            'derivable': 0,
            'not_derivable': 0,
        },
        'failed_apis': Counter(),
        'success_apis': Counter(),
        'errors': [],
        'sample_results': [],
    }
    
    completed = 0
    
    print(f"开始并行检查 ({max_workers} 线程)...")
    
    # 使用线程池并行处理（网络 I/O 密集型用线程更合适）
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_sample = {
            executor.submit(dynamic_checker.check_sample, sample): sample
            for sample in all_samples
        }
        
        # 收集结果
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            completed += 1
            
            try:
                sample_result = future.result()
            except Exception as e:
                sample_result = {
                    'sample_id': sample.sample_id,
                    'passed': False,
                    'errors': [str(e)],
                    'api_calls': [],
                }
            
            results_data['sample_results'].append(sample_result)
            
            if sample_result.get('passed'):
                results_data['passed_samples'] += 1
            else:
                results_data['failed_samples'] += 1
                if sample_result.get('errors'):
                    results_data['errors'].append({
                        'sample_id': sample_result.get('sample_id', 'unknown'),
                        'errors': sample_result['errors']
                    })
            
            # 统计 API 调用
            for call_result in sample_result.get('api_calls', []):
                results_data['api_call_stats']['total'] += 1
                if call_result.get('success'):
                    results_data['api_call_stats']['success'] += 1
                    results_data['success_apis'][call_result.get('api_name', 'unknown')] += 1
                elif call_result.get('error') == 'API not found in toolenv mapping':
                    results_data['api_call_stats']['not_found'] += 1
                else:
                    results_data['api_call_stats']['failed'] += 1
                    results_data['failed_apis'][call_result.get('api_name', 'unknown')] += 1
            
            # 统计 finish
            finish_type = sample_result.get('finish_type')
            if finish_type in results_data['finish_stats']:
                results_data['finish_stats'][finish_type] += 1
            
            # 统计 derivability
            deriv = sample_result.get('derivability')
            if deriv:
                results_data['derivability']['total_checked'] += 1
                if deriv.get('derivable'):
                    results_data['derivability']['derivable'] += 1
                else:
                    results_data['derivability']['not_derivable'] += 1
            
            # 进度
            if progress_interval and completed % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                pass_rate = results_data['passed_samples'] / completed if completed > 0 else 0
                print(f"  [{completed:,}/{total_samples:,}] {rate:.1f} 条/秒, 通过率: {pass_rate:.2%}")
    
    elapsed = time.time() - start_time
    
    # 计算 pass rate
    results_data['pass_rate'] = (
        results_data['passed_samples'] / results_data['total_samples'] * 100
        if results_data['total_samples'] > 0 else 0.0
    )
    
    # 计算 derivability rate
    if results_data['derivability']['total_checked'] > 0:
        results_data['derivability']['rate'] = (
            results_data['derivability']['derivable'] / 
            results_data['derivability']['total_checked'] * 100
        )
    else:
        results_data['derivability']['rate'] = 0.0
    
    # 汇总结果
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'max_workers': max_workers,
        
        # 样本统计
        'total_samples': total_samples,
        'passed_samples': results_data['passed_samples'],
        'failed_samples': results_data['failed_samples'],
        'pass_rate': results_data['pass_rate'],
        
        # API 调用统计
        'api_call_stats': results_data['api_call_stats'],
        
        # Finish 统计
        'finish_stats': results_data['finish_stats'],
        
        # 可推导性统计
        'derivability': results_data['derivability'],
        
        # 失败的 API 统计
        'failed_apis': dict(results_data['failed_apis'].most_common(50)),
        'success_apis': dict(results_data['success_apis'].most_common(50)),
        
        # 错误列表（完整保存）
        'errors': results_data['errors'],
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
    print()
    print("=" * 70)
    print(f"评估完成！耗时 {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print("=" * 70)
    print()
    print(f"总样本数: {total_samples:,}")
    print(f"通过: {results_data['passed_samples']:,} ({results_data['pass_rate']:.2f}%)")
    print(f"失败: {results_data['failed_samples']:,}")
    print(f"处理速度: {total_samples / elapsed:.1f} 条/秒")
    print()
    
    print("【API 调用统计】")
    api_stats = results_data['api_call_stats']
    print(f"  总调用: {api_stats['total']:,}")
    print(f"  成功: {api_stats['success']:,}")
    print(f"  失败: {api_stats['failed']:,}")
    print(f"  未找到: {api_stats['not_found']:,}")
    print()
    
    print("【Finish 统计】")
    finish_stats = results_data['finish_stats']
    print(f"  give_answer: {finish_stats['give_answer']:,}")
    print(f"  give_up_and_restart: {finish_stats['give_up_and_restart']:,}")
    print(f"  invalid: {finish_stats.get('invalid', 0):,}")
    print()
    
    deriv = results_data['derivability']
    if deriv['total_checked'] > 0:
        print("【可推导性统计】")
        print(f"  检查数: {deriv['total_checked']:,}")
        print(f"  可推导: {deriv['derivable']:,} ({deriv['rate']:.2f}%)")
        print()
    
    # 显示错误类型统计
    if results_data['errors']:
        print("【错误类型统计】")
        error_types = {}
        for err_item in results_data['errors']:
            if isinstance(err_item, dict):
                for err in err_item.get('errors', []):
                    err_type = str(err).split(':')[0] if ':' in str(err) else str(err)[:50]
                    error_types[err_type] = error_types.get(err_type, 0) + 1
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1])[:20]:
            print(f"  {err_type}: {count:,}")
        print()
    
    if results_data['failed_apis']:
        print("【失败最多的 API (Top 10)】")
        for api, count in results_data['failed_apis'].most_common(10):
            print(f"  {api}: {count}")
        print()
    
    return results


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Dynamic Executability 指标评估")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["toolbench"],
                        help="数据集名称（目前只支持 toolbench）")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数（用于测试，默认全量）")
    parser.add_argument("--toolenv-path", type=str, default=None,
                        help="ToolBench toolenv 路径")
    parser.add_argument("--rapidapi-key", type=str, default=None,
                        help="RapidAPI Key（也可通过 RAPIDAPI_KEY 环境变量设置）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--timeout", type=int, default=30,
                        help="API 调用超时时间（秒）")
    parser.add_argument("--workers", type=int, default=16,
                        help="并行线程数（默认 16）")
    parser.add_argument("--progress-interval", type=int, default=100,
                        help="进度显示间隔")
    
    args = parser.parse_args()
    
    # 获取 RapidAPI Key
    rapidapi_key = args.rapidapi_key or os.environ.get('RAPIDAPI_KEY')
    if not rapidapi_key:
        print("错误：需要 RapidAPI Key")
        print("请通过 --rapidapi-key 参数或 RAPIDAPI_KEY 环境变量提供")
        exit(1)
    
    # 根据数据集选择 loader 和 checker
    if args.dataset == "toolbench":
        from loaders import ToolBenchLoader
        from toolbench_executor import ToolBenchDynamicChecker
        
        loader = ToolBenchLoader(
            '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/toolbench_official/toolllama_G123_dfs_train.json'
        )
        
        toolenv_path = args.toolenv_path or '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/toolbench_official/toolenv/tools'
        checker = ToolBenchDynamicChecker(
            rapidapi_key=rapidapi_key,
            toolenv_path=toolenv_path,
            timeout=args.timeout,
        )
        dataset_name = "ToolBench"
    
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
        max_workers=args.workers,
    )
