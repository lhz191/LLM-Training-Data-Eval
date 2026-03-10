#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text GUI Agent Eval Metrics

单轮指标：
- format_check: 格式检查
- static_executability: 静态可执行性
- dynamic_executability: 动态可执行性
- diversity: 多样性评估
- task_complexity: 任务复杂度
- trajectory_validity: 轨迹有效性
- html_retention: HTML 保留率
- trustworthy: 安全可信性（Guard 模型）
"""

from .static_executability import compute_static_executability
from .dynamic_executability import compute_dynamic_executability
from .trustworthy import compute_trustworthy, compute_trustworthy_parallel

__all__ = [
    'compute_static_executability',
    'compute_dynamic_executability',
    'compute_trustworthy',
    'compute_trustworthy_parallel',
]

