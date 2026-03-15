#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Quality 指标 - 报告内容质量评估

通过传入数据集特有的 QualityChecker 来评估报告质量。
不同数据集有不同的评估维度和方式，框架动态收集各 checker 返回的维度名和分数。

已实现的 checker 及其评估维度：

- IU X-Ray:    医学术语规范性(terminology)、临床逻辑性(clinical_logic)、信息密度(information_density)
- ShareGPT4V:  描述丰富度(richness)、训练价值(training_value)、内容连贯性(coherence)

各 checker 自行决定使用 sample 的哪些字段：
- IUXRayQualityChecker: 只用 report（所有样本同一 instruction，无需传入）
- ShareGPT4VQualityChecker: 用 instruction + report（instruction 有实际意义）

使用方式:
    from loaders import IUXRayLoader
    from executor.iu_xray import IUXRayQualityChecker
    from metrics.report_quality import compute_report_quality

    loader = IUXRayLoader('/path/to/IU-Xray', split='train')
    checker = IUXRayQualityChecker()

    results = compute_report_quality(
        data_iterator=loader.iterate(),
        quality_checker=checker,
        dataset_name='IU X-Ray (train)',
        output_file='results/quality_iu_xray.json',
        max_samples=100,
    )
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Optional, Iterator, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import ImageToReportSample
from report_executor import QualityChecker


def compute_report_quality(
    data_iterator: Iterator[ImageToReportSample],
    quality_checker: QualityChecker,
    dataset_name: str = "unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 10,
    parallel: bool = False,
    max_workers: int = 8,
) -> Dict[str, Any]:
    """
    计算报告质量指标

    Args:
        data_iterator: ImageToReportSample 迭代器
        quality_checker: 数据集特有的质量评估器
        dataset_name: 数据集名称
        output_file: 结果输出 JSON 路径
        max_samples: 最大样本数（LLM judge 较慢，建议先小批量测试）
        progress_interval: 进度显示间隔
        parallel: 是否并行（多线程并发调用 LLM）
        max_workers: 并行线程数

    Returns:
        结果字典
    """
    mode_str = "并行" if parallel else "串行"
    print("=" * 70)
    print(f"Report Quality Evaluation - {mode_str}")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"评估器: {quality_checker.__class__.__name__}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    print()

    start_time = time.time()

    # 收集样本
    all_samples = []
    for sample in data_iterator:
        if max_samples and len(all_samples) >= max_samples:
            break
        all_samples.append(sample)

    total_to_process = len(all_samples)
    print(f"共 {total_to_process:,} 条样本待处理")
    print()

    # 统计
    total = 0
    passed_count = 0
    llm_failures = 0

    # 各维度累计分数（动态收集，不预设维度名）
    dim_sums: Dict[str, float] = {}
    dim_counts: Dict[str, int] = {}

    sample_results: List[Dict[str, Any]] = []

    def _process_one(sample: ImageToReportSample) -> Dict[str, Any]:
        result = quality_checker.check(sample)
        result["sample_id"] = sample.sample_id
        return result

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_one, s): s.sample_id
                for s in all_samples
            }
            for future in as_completed(futures):
                total += 1
                result = future.result()
                sample_results.append(result)
                _accumulate(result, dim_sums, dim_counts)
                if result.get("passed", False):
                    passed_count += 1
                if result.get("llm_error", False):
                    llm_failures += 1
                if progress_interval and total % progress_interval == 0:
                    _print_progress(total, total_to_process, start_time, dim_sums, dim_counts, passed_count)
    else:
        for sample in all_samples:
            total += 1
            result = _process_one(sample)
            sample_results.append(result)
            _accumulate(result, dim_sums, dim_counts)
            if result.get("passed", False):
                passed_count += 1
            if result.get("llm_error", False):
                llm_failures += 1
            if progress_interval and total % progress_interval == 0:
                _print_progress(total, total_to_process, start_time, dim_sums, dim_counts, passed_count)

    elapsed = time.time() - start_time

    # 计算平均分
    avg_scores = {
        dim: dim_sums[dim] / dim_counts[dim]
        for dim in dim_sums if dim_counts.get(dim, 0) > 0
    }
    overall_avg = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0.0
    pass_rate = passed_count / total if total > 0 else 0.0

    results = {
        "dataset": dataset_name,
        "checker": quality_checker.__class__.__name__,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 2),

        "total": total,
        "passed": passed_count,
        "pass_rate": pass_rate,
        "llm_failures": llm_failures,

        "avg_scores": avg_scores,
        "overall_avg": overall_avg,

        "sample_results": sample_results,
    }

    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")

    # 打印摘要
    print()
    print("=" * 70)
    print(f"评估完成！耗时 {elapsed:.1f} 秒")
    print("=" * 70)
    print()
    print(f"  总样本数:     {total:,}")
    print(f"  通过质量门槛: {passed_count:,} ({pass_rate:.2%})")
    print(f"  LLM 失败:     {llm_failures}")
    print()
    print(f"【各维度平均分】(1-5)")
    for dim, score in sorted(avg_scores.items()):
        print(f"  {dim}: {score:.2f}")
    print(f"  ----")
    print(f"  Overall: {overall_avg:.2f}")
    print()

    # 分数分布
    if sample_results:
        _print_score_distribution(sample_results)

    print("=" * 70)

    return results


# =============================================================================
# 辅助函数
# =============================================================================

def _accumulate(result: Dict, dim_sums: Dict[str, float], dim_counts: Dict[str, int]):
    """累加各维度分数"""
    scores = result.get("scores", {})
    for dim, score in scores.items():
        dim_sums[dim] = dim_sums.get(dim, 0.0) + score
        dim_counts[dim] = dim_counts.get(dim, 0) + 1


def _print_progress(total, total_to_process, start_time, dim_sums, dim_counts, passed_count):
    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0
    pass_rate = passed_count / total if total > 0 else 0
    avgs = " | ".join(
        f"{dim}: {dim_sums[dim] / dim_counts[dim]:.2f}"
        for dim in sorted(dim_sums) if dim_counts.get(dim, 0) > 0
    )
    print(f"  [{total:,}/{total_to_process:,}] {rate:.1f} 条/秒 | 通过: {pass_rate:.2%} | {avgs}")


def _print_score_distribution(sample_results: List[Dict]):
    """打印分数区间分布"""
    all_avgs = [r.get("avg_score", 0.0) for r in sample_results if not r.get("llm_error", False)]
    if not all_avgs:
        return

    n = len(all_avgs)
    bins = [(4.5, 5.0, "优秀"), (3.5, 4.5, "良好"), (2.5, 3.5, "一般"), (1.5, 2.5, "较差"), (0.0, 1.5, "极差")]
    print(f"  综合分分布 (N={n}):")
    for lo, hi, label in bins:
        count = sum(1 for s in all_avgs if lo <= s <= hi)
        if count > 0:
            print(f"    {label} ({lo}-{hi}): {count}/{n} ({100 * count / n:.1f}%)")
    print()
