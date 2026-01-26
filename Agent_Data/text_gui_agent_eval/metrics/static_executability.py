#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static Executability 指标 - GUI Agent 静态可执行性检查

基于 MHTML/HTML 快照检查 GUI Agent 数据集样本的可执行性：
- 坐标定位成功率：使用 bounding_box 坐标能否定位到目标元素
- 属性定位成功率：使用 tag/class/id 等属性能否定位到目标元素

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
        - coord_rate: 坐标定位成功率
        - attr_rate: 属性定位成功率
        - 详细的 record 和 action 级别结果
    """
    print("=" * 70)
    print("Static Executability Evaluation")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    # Record 级别统计
    total_records = 0
    records_with_errors = 0
    records_with_warnings = 0
    
    # Action 级别统计
    total_actions = 0
    verified_actions = 0  # 有 MHTML 可验证的
    coord_success = 0
    attr_success = 0
    
    # 详细结果
    record_results = []
    error_records = []
    
    for record in data_iterator:
        if max_samples and total_records >= max_samples:
            break
        
        total_records += 1
        
        # 检查可执行性
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
        
        # Action 级别统计
        total_actions += stats.get('total_actions', 0)
        verified_actions += stats.get('verified_actions', 0)
        coord_success += stats.get('coord_success', 0)
        attr_success += stats.get('attr_success', 0)
        
        # 记录详细结果
        record_results.append({
            'sample_id': record.sample_id,
            'annotation_id': record.metadata.get('annotation_id', ''),
            'website': record.website,
            'total_actions': stats.get('total_actions', 0),
            'verified_actions': stats.get('verified_actions', 0),
            'coord_success': stats.get('coord_success', 0),
            'attr_success': stats.get('attr_success', 0),
            'coord_rate': stats.get('coord_rate', 0.0),
            'attr_rate': stats.get('attr_rate', 0.0),
            'errors': errors,
            'warnings': warnings,
        })
        
        # 进度
        if progress_interval and total_records % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = total_records / elapsed if elapsed > 0 else 0
            curr_coord_rate = coord_success / verified_actions if verified_actions > 0 else 0
            curr_attr_rate = attr_success / verified_actions if verified_actions > 0 else 0
            print(f"  [{total_records:,}] {rate:.2f} 条/秒")
            print(f"    坐标定位: {coord_success}/{verified_actions} ({curr_coord_rate:.1%})")
            print(f"    属性定位: {attr_success}/{verified_actions} ({curr_attr_rate:.1%})")
    
    elapsed = time.time() - start_time
    
    # 计算总体成功率
    coord_rate = coord_success / verified_actions if verified_actions > 0 else 0.0
    attr_rate = attr_success / verified_actions if verified_actions > 0 else 0.0
    
    # 结果
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        
        # Record 级别统计
        'total_records': total_records,
        'records_with_errors': records_with_errors,
        'records_with_warnings': records_with_warnings,
        
        # Action 级别统计
        'total_actions': total_actions,
        'verified_actions': verified_actions,
        'coord_success': coord_success,
        'attr_success': attr_success,
        
        # 成功率
        'coord_rate': coord_rate,
        'attr_rate': attr_rate,
        
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
    
    # 打印摘要
    _print_summary(results, elapsed)
    
    return results


def _print_summary(results: Dict[str, Any], elapsed: float):
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
    print(f"  可验证 Action 数: {results['verified_actions']:,}")
    print()
    print(f"【静态可执行性】")
    print(f"  坐标定位成功率: {results['coord_success']:,}/{results['verified_actions']:,} "
          f"({results['coord_rate']:.2%})")
    print(f"  属性定位成功率: {results['attr_success']:,}/{results['verified_actions']:,} "
          f"({results['attr_rate']:.2%})")
    print()
    
    # 按网站统计
    website_stats = {}
    for r in results['record_results']:
        site = r.get('website') or 'unknown'
        if site not in website_stats:
            website_stats[site] = {
                'records': 0,
                'verified': 0,
                'coord': 0,
                'attr': 0,
            }
        website_stats[site]['records'] += 1
        website_stats[site]['verified'] += r.get('verified_actions', 0)
        website_stats[site]['coord'] += r.get('coord_success', 0)
        website_stats[site]['attr'] += r.get('attr_success', 0)
    
    if len(website_stats) > 1:
        print(f"【按网站统计】")
        sorted_sites = sorted(website_stats.items(), key=lambda x: -x[1]['records'])
        for site, stats in sorted_sites[:10]:
            v = stats['verified']
            if v > 0:
                coord_r = stats['coord'] / v
                attr_r = stats['attr'] / v
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
    
    parser = argparse.ArgumentParser(description="静态可执行性指标评估")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["mind2web", "weblinx"],
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
