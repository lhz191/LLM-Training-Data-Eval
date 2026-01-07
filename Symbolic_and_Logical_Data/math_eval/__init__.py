#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Reasoning Data Evaluation

统一的数学推理数据评估框架，支持：
- 数据加载：MetaMathQA, OpenMathInstruct-1, GSM8K-Aug
- 答案验证：精确匹配、数值匹配、符号匹配
- 代码执行：可选的 Python 代码执行验证

Usage:
    from math_eval import load_dataset, MathSample
    
    # 加载数据集
    samples = load_dataset("metamathqa", "/path/to/MetaMathQA", limit=100)
    
    # 查看样本
    print(samples[0])
"""

from .data_types import MathSample
from .loaders import (
    load_dataset,
    get_supported_datasets,
    MetaMathQALoader,
    OpenMathInstructLoader,
    GSM8KAugLoader,
)

__version__ = "0.1.0"
__all__ = [
    # 数据类型
    "MathSample",
    # 加载器
    "load_dataset",
    "get_supported_datasets",
    "MetaMathQALoader",
    "OpenMathInstructLoader",
    "GSM8KAugLoader",
]

