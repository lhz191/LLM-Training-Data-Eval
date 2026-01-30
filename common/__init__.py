#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Training Data Evaluation - Common Module

统一基类和注册表，用于连接不同模态的评估器。

使用方式：
    from common import BaseFormatChecker, BaseExecutabilityChecker
    from common import register_checker, get_checker, list_checkers
"""

from .base import (
    # 根基类
    BaseChecker,
    # 格式检查
    BaseFormatChecker,
    # 可执行性检查
    BaseExecutabilityChecker,
    BaseDynamicChecker,
    # 代码相关
    BaseCodeExtractor,
    BaseCodeExecutor,
    BaseResultComparator,
    # 元素定位
    BaseHTMLLocator,
    # 答案提取
    BaseAnswerExtractor,
)

from .registry import (
    register_checker,
    get_checker,
    list_checkers,
    list_modalities,
    print_registry,
)

__version__ = "0.1.0"

__all__ = [
    # 基类
    "BaseChecker",
    "BaseFormatChecker",
    "BaseExecutabilityChecker",
    "BaseDynamicChecker",
    "BaseCodeExtractor",
    "BaseCodeExecutor",
    "BaseResultComparator",
    "BaseHTMLLocator",
    "BaseAnswerExtractor",
    # 注册表
    "register_checker",
    "get_checker",
    "list_checkers",
    "list_modalities",
]
