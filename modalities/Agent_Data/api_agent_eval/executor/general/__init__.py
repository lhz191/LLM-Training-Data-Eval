#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用执行器（General）

基于 data_types.py 合同 (APIAgentSample / ToolDefinition / APICall / Parameter)
的通用检查器，不依赖任何数据集特有逻辑。

提供:
  - GeneralFormatChecker（格式检查）
    纯粹基于 dataclass 字段定义进行检查：
    字段是否存在、类型是否正确、工具名是否唯一、参数名是否重复等。

  - GeneralExecutabilityChecker（静态可执行性检查）
    基于合同进行 API 调用的静态可执行性验证：
    调用的工具是否存在于工具列表、参数名是否匹配、必需参数是否缺失等。

适用于任何已通过 Loader 转换为 APIAgentSample 的数据，或用户按合同自建的数据。
"""

from .GeneralFormatChecker import GeneralFormatChecker
from .GeneralExecutabilityChecker import GeneralExecutabilityChecker

__all__ = [
    'GeneralFormatChecker',
    'GeneralExecutabilityChecker',
]
