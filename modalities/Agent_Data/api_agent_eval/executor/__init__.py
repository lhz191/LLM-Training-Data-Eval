#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Agent 执行器包

按数据集组织的执行器模块：
- executor.general:     通用检查器（基于 data_types.py 合同，不依赖特定数据集）
- executor.toolbench:   ToolBench 数据集执行器
- executor.xlam:        xLAM 数据集执行器
- executor.arcee_agent: Arcee Agent Data 执行器

使用方式：
    # 方式1：通用检查器（适用于任何符合 data_types.py 合同的数据）
    from executor.general import GeneralFormatChecker, GeneralExecutabilityChecker
    checker = GeneralFormatChecker()
    errors, warnings = checker.check(sample)

    # 方式2：直接导入特定数据集检查器
    from executor.toolbench import ToolBenchFormatChecker
    from executor.xlam import XLAMFormatChecker

    # 方式3：通过全局注册表
    from api_executor import get_format_checker
    checker = get_format_checker('general')   # 通用
    checker = get_format_checker('toolbench') # ToolBench 特有
"""

import os
import sys

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from api_executor import (
    register_format_checker,
    register_executability_checker,
    register_dynamic_checker,
)


# =============================================================================
# 导入并注册 General (通用) 检查器
# =============================================================================

from .general import (
    GeneralFormatChecker,
    GeneralExecutabilityChecker,
)

register_format_checker('general', GeneralFormatChecker)
register_executability_checker('general', GeneralExecutabilityChecker)


# =============================================================================
# 导入并注册 ToolBench 检查器
# =============================================================================

from .toolbench import (
    ToolBenchFormatChecker,
    ToolBenchExecutabilityChecker,
    ToolBenchDynamicChecker,
)

register_format_checker('toolbench', ToolBenchFormatChecker)
register_executability_checker('toolbench', ToolBenchExecutabilityChecker)
register_dynamic_checker('toolbench', ToolBenchDynamicChecker)


# =============================================================================
# 导入并注册 xLAM 检查器
# =============================================================================

from .xlam import (
    XLAMFormatChecker,
    XLAMExecutabilityChecker,
)

register_format_checker('xlam', XLAMFormatChecker)
register_format_checker('xlam-60k', XLAMFormatChecker)  # 别名
register_executability_checker('xlam', XLAMExecutabilityChecker)
register_executability_checker('xlam-60k', XLAMExecutabilityChecker)  # 别名


# =============================================================================
# 导入并注册 Arcee Agent 检查器
# =============================================================================

from .arcee_agent import (
    ArceeAgentFormatChecker,
    ArceeAgentExecutabilityChecker,
)

register_format_checker('arcee-agent', ArceeAgentFormatChecker)
register_format_checker('arcee-agent-data', ArceeAgentFormatChecker)  # 别名
register_executability_checker('arcee-agent', ArceeAgentExecutabilityChecker)
register_executability_checker('arcee-agent-data', ArceeAgentExecutabilityChecker)  # 别名


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # General (通用)
    'GeneralFormatChecker',
    'GeneralExecutabilityChecker',
    # ToolBench
    'ToolBenchFormatChecker',
    'ToolBenchExecutabilityChecker',
    'ToolBenchDynamicChecker',
    # xLAM
    'XLAMFormatChecker',
    'XLAMExecutabilityChecker',
    # Arcee Agent
    'ArceeAgentFormatChecker',
    'ArceeAgentExecutabilityChecker',
]
