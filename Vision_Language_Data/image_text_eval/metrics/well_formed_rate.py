#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Well-Formed Rate (WFR) 指标计算

WFR = (1/|M_gen|) * Σ V(m, S)
其中 V(m, S) ∈ {0, 1} 指示生成结果 m 是否符合 schema S。

使用方式:
    from loaders import GeneralLoader
    from image_executor import get_format_checker
    from metrics.well_formed_rate import compute_well_formed_rate
    
    loader = GeneralLoader('/path/to/data.jsonl')
    checker = get_format_checker('coco_caption', strict=True)
    
    score = compute_well_formed_rate(data_iterator=loader.iterate(), checker=checker)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Iterator

from data_types import ImageTextSample
from image_executor import FormatChecker


def compute_well_formed_rate(
    data_iterator: Iterator[ImageTextSample],
    checker: FormatChecker,
) -> float:
    """
    计算 Well-Formed Rate (WFR)
    
    Args:
        data_iterator: ImageTextSample 迭代器
        checker: 格式检查器实例
    
    Returns:
        WFR（0-1 的 float）
    """
    samples = list(data_iterator)
    
    if not samples:
        return 0.0

    valid_count = sum(
        1 for sample in samples
        if checker.check(sample)
    )

    wfr = valid_count / len(samples)
    return wfr
