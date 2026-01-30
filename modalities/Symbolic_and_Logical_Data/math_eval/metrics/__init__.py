#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标模块

包含以下指标的计算入口：
- format_check: 格式检查
- validity: 代码有效性检查
- reasoning_validity: 推理有效性检查
- faithfulness: 答案忠实性检查
- diversity: 多样性指标（Vendi Score / KNN）
"""

from .format_check import compute_format_check
from .validity import compute_validity
from .reasoning_validity import compute_reasoning_validity
from .faithfulness import compute_faithfulness
from .diversity import compute_diversity

__all__ = [
    'compute_format_check',
    'compute_validity',
    'compute_reasoning_validity',
    'compute_faithfulness',
    'compute_diversity',
]
