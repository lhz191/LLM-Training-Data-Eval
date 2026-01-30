#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LILA 格式检查器
"""

import os
import sys
from typing import List, Tuple

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from code_executor import FormatChecker

class LILAFormatChecker(FormatChecker):
    """
    LILA 数据集格式检查器
    
    基于论文 Section 3.1 的统一格式要求：
    
    必需字段检查：
    1. question (Input): 问题文本（不能为空）
    2. solution (Output Program): Python 程序列表（不能为空）
    3. ground_truth (Output Answer): 答案列表（不能为空）
    """
    
    def check(self, sample) -> Tuple[List[str], List[str]]:
        """检查 LILA 样本的格式正确性"""
        errors = []
        warnings = []
        
        # === 必需字段检查 ===
        if not sample.question:
            errors.append("Missing or empty 'question' (Input) field")
        
        if not sample.solution:
            errors.append("Missing or empty 'solution' (Output Program) field")
        elif isinstance(sample.solution, list):
            # 检查程序列表是否有空元素
            for i, prog in enumerate(sample.solution):
                if not prog or not str(prog).strip():
                    errors.append(f"Empty program at index {i}")
        
        if sample.ground_truth is None:
            errors.append("Missing 'ground_truth' (Output Answer) field")
        elif isinstance(sample.ground_truth, list) and len(sample.ground_truth) == 0:
            errors.append("Empty 'ground_truth' (Output Answer) list")
        elif isinstance(sample.ground_truth, str) and not sample.ground_truth.strip():
            errors.append("Empty 'ground_truth' (Output Answer) field")
        
        return errors, warnings
