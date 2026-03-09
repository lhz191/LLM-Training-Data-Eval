#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标模块

包含以下指标的计算入口：
- format_check: 格式检查
- executability: 静态可执行性检查
- dynamic_executability: 动态可执行性检查（真实 API 调用）
- diversity: 多样性指标（Vendi Score / KNN / API Call Diversity）
- task_complexity: 任务复杂度评估
- trustworthy: 安全可信性评估（Guard 模型）

生成数据特有指标（synthetic_specific/）：
- template_consistency: 模板一致性
- template_divergence: 模板偏离度
- distribution_similarity: 分布相似度 (MMD)
"""

from .format_check import compute_format_check
from .executability import compute_executability
from .dynamic_executability import compute_dynamic_executability
from .diversity import compute_diversity
from .task_complexity import compute_task_complexity
from .trustworthy import compute_trustworthy

__all__ = [
    'compute_format_check',
    'compute_executability',
    'compute_dynamic_executability',
    'compute_diversity',
    'compute_task_complexity',
    'compute_trustworthy',
]
