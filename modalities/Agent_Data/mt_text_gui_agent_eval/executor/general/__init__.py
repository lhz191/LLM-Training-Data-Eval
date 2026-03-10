#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用多轮执行器（General）

基于 Session / Record / Action 合同的通用检查器，不依赖任何数据集特有逻辑。

提供:
  - GeneralSessionFormatChecker（Session 格式检查）
"""

from .GeneralSessionFormatChecker import GeneralSessionFormatChecker

__all__ = [
    'GeneralSessionFormatChecker',
]
