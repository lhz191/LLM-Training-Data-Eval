#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Agent 执行器包

按数据集组织的执行器模块：
- executor.toolbench: ToolBench 数据集执行器
- executor.xlam: xLAM 数据集执行器

使用方式：
    # 方式1：直接导入类
    from executor.toolbench import ToolBenchFormatChecker
    from executor.xlam import XLAMFormatChecker
    
    # 方式2：通过全局注册表
    from api_executor import get_format_checker
    checker = get_format_checker('toolbench')
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
# 导出
# =============================================================================

__all__ = [
    # ToolBench
    'ToolBenchFormatChecker',
    'ToolBenchExecutabilityChecker',
    'ToolBenchDynamicChecker',
    # xLAM
    'XLAMFormatChecker',
    'XLAMExecutabilityChecker',
]
