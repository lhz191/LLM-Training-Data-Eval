#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math 执行器包

按数据集组织的执行器模块：
- executor.lila: LILA 数据集执行器
- executor.openmath: OpenMath 数据集执行器

使用方式：
    # 方式1：直接导入类
    from executor.lila import LILACodeExecutor
    from executor.openmath import OpenMathExecutor
    
    # 方式2：通过全局注册表
    from code_executor import get_executor
    executor = get_executor('lila')
"""

import os
import sys

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from code_executor import (
    register_answer_extractor,
    register_code_extractor,
    register_executor,
)


# =============================================================================
# 导入并注册 LILA
# =============================================================================

from .lila import (
    LILACodeExtractor,
    LILACodeExecutor,
    LILAResultComparator,
    LILAFormatChecker,
)

register_code_extractor('lila', LILACodeExtractor)
register_executor('lila', LILACodeExecutor)


# =============================================================================
# 导入并注册 OpenMath
# =============================================================================

from .openmath import (
    BoxedAnswerExtractor,
    DirectAnswerExtractor,
    OpenMathCodeExtractor,
    OpenMathExecutor,
    OpenMathExecutorFast,
    OpenMathResultComparator,
    OpenMathFormatChecker,
)

# 答案提取器
register_answer_extractor('boxed', BoxedAnswerExtractor)
register_answer_extractor('openmath', BoxedAnswerExtractor)
register_answer_extractor('openmathinstruct1', BoxedAnswerExtractor)
register_answer_extractor('direct', DirectAnswerExtractor)

# 代码提取器
register_code_extractor('openmath', OpenMathCodeExtractor)
register_code_extractor('openmathinstruct1', OpenMathCodeExtractor)

# 执行器
register_executor('openmath', OpenMathExecutor)
register_executor('openmathinstruct1', OpenMathExecutor)
register_executor('openmathfast', OpenMathExecutorFast)
register_executor('openmathinstruct1fast', OpenMathExecutorFast)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # LILA
    'LILACodeExtractor',
    'LILACodeExecutor',
    'LILAResultComparator',
    'LILAFormatChecker',
    # OpenMath
    'BoxedAnswerExtractor',
    'DirectAnswerExtractor',
    'OpenMathCodeExtractor',
    'OpenMathExecutor',
    'OpenMathExecutorFast',
    'OpenMathResultComparator',
    'OpenMathFormatChecker',
]
