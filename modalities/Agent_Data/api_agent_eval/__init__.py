#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Agent Data Evaluation

统一的 API Agent 数据评估框架，支持：
- 数据加载：ToolBench, xLAM-60k
- 格式检查：数据结构验证
- 可执行性检查：静态 + 动态
- 多样性评估：Vendi Score / KNN

Usage:
    from api_agent_eval import APIAgentSample, ToolBenchLoader, XLAMLoader
    
    # 加载数据集
    loader = ToolBenchLoader('/path/to/toolbench.json')
    samples = loader.parse_all()
    
    # 查看样本
    print(samples[0])
"""

from .data_types import APIAgentSample, ToolDefinition, APICall, Parameter
from .loaders import ToolBenchLoader, XLAMLoader, load_toolbench, load_xlam
from .api_executor import (
    FormatChecker,
    ExecutabilityChecker,
    DynamicChecker,
    get_format_checker,
    get_executability_checker,
    get_dynamic_checker,
    list_format_checkers,
    list_executability_checkers,
    list_dynamic_checkers,
)

__version__ = "0.1.0"
__all__ = [
    # 数据类型
    "APIAgentSample",
    "ToolDefinition",
    "APICall",
    "Parameter",
    # 加载器
    "ToolBenchLoader",
    "XLAMLoader",
    "load_toolbench",
    "load_xlam",
    # 检查器基类
    "FormatChecker",
    "ExecutabilityChecker",
    "DynamicChecker",
    # 工厂函数
    "get_format_checker",
    "get_executability_checker",
    "get_dynamic_checker",
    "list_format_checkers",
    "list_executability_checkers",
    "list_dynamic_checkers",
]
