#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Format Check 指标 - Image-to-Report 数据格式验证

检查样本的结构完整性：必需字段存在性、<image> token 匹配、数据集特有约束等。

使用方式:
    from loaders import IUXRayLoader
    from executor import IUXRayFormatChecker
    from metrics.format_check import compute_format_check

    loader = IUXRayLoader('/path/to/IU-Xray', split='train')
    checker = IUXRayFormatChecker()

    results = compute_format_check(
        data_iterator=loader.iterate(),
        format_checker=checker,
        dataset_name='IU X-Ray (train)',
        output_file='results/format_check_iu_xray.json',
    )
"""

import json
import time
import os
from datetime import datetime
from typing import Optional, Iterator, Dict, List, Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_types import ImageToReportSample
from report_executor import FormatChecker


def compute_format_check(
    data_iterator: Iterator[ImageToReportSample],
    format_checker: FormatChecker,
    dataset_name: str = "unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 10000,
) -> Dict[str, Any]:
    """
    计算 Format Check 指标

    Args:
        data_iterator: ImageToReportSample 迭代器
        format_checker: 格式检查器实例
        dataset_name: 数据集名称（用于输出）
        output_file: 结果输出 JSON 路径（可选）
        max_samples: 最大样本数（用于测试，None 表示全量）
        progress_interval: 每处理多少条打印一次进度

    Returns:
        结果字典，包含统计和错误样本详情
    """
    print("=" * 70)
    print("Format Check Evaluation")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = time.time()

    total = 0
    passed = 0
    with_errors = 0
    with_warnings = 0

    error_samples = []
    warning_samples = []

    for sample in data_iterator:
        if max_samples and total >= max_samples:
            break

        total += 1

        errors, warnings = format_checker.check(sample)

        if errors:
            with_errors += 1
            error_samples.append({
                "sample_id": sample.sample_id,
                "errors": errors,
                "warnings": warnings,
            })
        elif warnings:
            with_warnings += 1
            passed += 1
            warning_samples.append({
                "sample_id": sample.sample_id,
                "warnings": warnings,
            })
        else:
            passed += 1

        if progress_interval and total % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            pass_rate = passed / total if total > 0 else 0
            print(f"  [{total:,}] {rate:.1f} 条/秒, 通过率: {pass_rate:.2%}")

    elapsed = time.time() - start_time
    pass_rate = passed / total if total > 0 else 0
    error_rate = with_errors / total if total > 0 else 0
    warning_rate = with_warnings / total if total > 0 else 0

    results = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "total": total,
        "passed": passed,
        "with_errors": with_errors,
        "with_warnings": with_warnings,
        "pass_rate": pass_rate,
        "error_rate": error_rate,
        "warning_rate": warning_rate,
        "error_samples": error_samples,
        "warning_samples": warning_samples,
    }

    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")

    # --- 打印摘要 ---
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

    if error_samples:
        print("【错误类型统计】")
        error_types: Dict[str, int] = {}
        for s in error_samples:
            for err in s["errors"]:
                err_type = err.split(":")[0] if ":" in err else err
                error_types[err_type] = error_types.get(err_type, 0) + 1
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1])[:20]:
            print(f"  {err_type}: {count:,}")
        print()

    if warning_samples:
        print("【警告类型统计】")
        warning_types: Dict[str, int] = {}
        for s in warning_samples:
            for warn in s["warnings"]:
                warn_type = warn.split(":")[0] if ":" in warn else warn
                warning_types[warn_type] = warning_types.get(warn_type, 0) + 1
        for warn_type, count in sorted(warning_types.items(), key=lambda x: -x[1])[:20]:
            print(f"  {warn_type}: {count:,}")
        print()

    return results
