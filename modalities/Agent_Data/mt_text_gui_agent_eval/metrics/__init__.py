#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮 Text-GUI Agent 评测指标

结构：
  metrics/
    diversity.py              ← 继承单轮，展平 + session-level 分析
    format_check.py           ← 继承单轮，展平后调用
    static_executability.py   ← 继承单轮，展平后调用
    task_complexity.py        ← 继承单轮，展平后调用
    trajectory_validity.py    ← 继承单轮，展平后调用
    html_retention.py         ← 继承单轮，展平后调用
    dynamic_executability.py  ← 继承单轮，展平后调用
    mt_specific/              ← 纯多轮专有指标（不继承单轮）
      cross_turn_consistency.py
      context_utilization.py
      generation_quality.py
"""

from .diversity import compute_diversity_multiturn
from .format_check import compute_format_check
from .static_executability import compute_static_executability
from .task_complexity import compute_task_complexity
from .trajectory_validity import compute_trajectory_validity
from .html_retention import compute_html_retention
from .dynamic_executability import compute_dynamic_executability

from .mt_specific import (
    compute_cross_turn_consistency,
    compute_context_utilization,
    compute_generation_quality,
    compute_generation_quality_multiturn,
)

__all__ = [
    # 继承单轮（展平后调用）
    'compute_diversity_multiturn',
    'compute_format_check',
    'compute_static_executability',
    'compute_task_complexity',
    'compute_trajectory_validity',
    'compute_html_retention',
    'compute_dynamic_executability',
    # 纯多轮专有
    'compute_cross_turn_consistency',
    'compute_context_utilization',
    'compute_generation_quality',
    'compute_generation_quality_multiturn',
]
