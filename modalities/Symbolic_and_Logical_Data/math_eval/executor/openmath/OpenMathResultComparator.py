#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenMath 结果比较器
"""

import os
import sys
from typing import Optional, Tuple, Any

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from code_executor import ResultComparator, compare_results

class OpenMathResultComparator(ResultComparator):
    """
    OpenMath 数据集结果比较器
    
    - result: 单个执行结果
    - expected: 单个预期值（来自 llm-code-output 或 ground_truth）
    """
    
    def compare(self, result: Any, expected: Any) -> bool:
        """
        OpenMath: 直接比较 result 和 expected
        """
        if expected is None:
            return True
        
        # 如果 expected 是列表，取第一个（OpenMath 通常只有一个答案）
        if isinstance(expected, list):
            if len(expected) == 0:
                return True
            expected = expected[0]
        
        return compare_results(result, expected)

