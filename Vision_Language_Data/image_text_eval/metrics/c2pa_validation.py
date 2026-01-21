#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C2PA Validation Rate 指标 - 内容来源可信度评估

基于 C2PA（Coalition for Content Provenance and Authenticity）标准，
评估图像是否具有可验证的来源凭证。

指标定义：
  C2PA_Validation_Rate = (1 / |M|) * ∑_{m∈M} V_c2pa(m)
  
  其中 V_c2pa(m) = 1 若图片包含完整且可验证的 C2PA 凭证，否则为 0

注意：
- 需要安装 c2patool: https://github.com/contentauth/c2pa-rs/releases
- 对于普通生成图像（如 Stable Diffusion），通常该值为 0

使用方式:
    from loaders import GeneralLoader
    from metrics.c2pa_validation import compute_c2pa_validation_rate
    
    loader = GeneralLoader('/path/to/data.jsonl')
    
    rate = compute_c2pa_validation_rate(data_iterator=loader.iterate(), tool_path='./c2patool')
"""

import os
import subprocess
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Iterator, Optional

from data_types import ImageTextSample


def verify_c2pa(image_path: str, tool_path: str = "./c2patool") -> bool:
    """
    Use official c2patool to verify C2PA provenance.
    Return True iff full validation passes.
    """
    if not os.path.exists(image_path):
        return False

    try:
        res = subprocess.run(
            [tool_path, "verify", image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        output = (res.stdout + res.stderr).lower()

        if "no c2pa manifest" in output:
            return False
        if "signature valid" in output and "manifest integrity valid" in output:
            return True
        return False
    except Exception:
        return False


def compute_c2pa_validation_rate(
    data_iterator: Iterator[ImageTextSample],
    tool_path: str = "./c2patool",
) -> float:
    """
    计算 C2PA Validation Rate（来源凭证验证率）
    
    Args:
        data_iterator: ImageTextSample 迭代器
        tool_path: c2patool 可执行文件路径
    
    Returns:
        C2PA 验证通过率（0-1 的 float）
    """
    samples = list(data_iterator)
    
    # 筛选有 image_path 的样本
    samples_with_image = [s for s in samples if s.image_path]
    total = len(samples_with_image)

    if total == 0:
        return 0.0

    validated = 0
    for sample in samples_with_image:
        if verify_c2pa(sample.image_path, tool_path):
            validated += 1

    rate = validated / total
    return rate
