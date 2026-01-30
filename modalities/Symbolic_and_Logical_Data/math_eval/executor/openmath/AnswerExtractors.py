#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenMath 答案提取器
"""

import os
import sys
import re
from typing import Optional, Tuple, Any, List

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from code_executor import AnswerExtractor

class BoxedAnswerExtractor(AnswerExtractor):
    """
    提取 \\boxed{} 格式的答案
    
    适用于: OpenMathInstruct-1, NuminaMath, MATH 等
    
    策略：
    1. 找最后一个 \\boxed（rfind）
    2. 递归提取最内层的 \\boxed 内容
    3. 清理首尾的 $ 符号和多余空格
    """
    
    def _extract_single_boxed(self, text: str) -> Optional[str]:
        """从文本中提取最后一个 \\boxed{} 的内容"""
        # 找最后一个 \boxed
        idx = text.rfind("\\boxed")
        if idx < 0:
            return None
        
        # 找到 { 开始的位置
        i = idx + len("\\boxed")
        while i < len(text) and text[i] != '{':
            i += 1
        if i >= len(text):
            return None
        
        # 括号匹配
        start = i + 1
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        
        if depth == 0:
            return text[start:i-1]
        return None
    
    def extract(self, solution: str) -> Optional[str]:
        """
        提取最内层的 \\boxed{} 答案
        
        处理嵌套情况如：
        - \\boxed{\\boxed{36\\pi}} → 36\\pi
        - \\boxed{-3+26=\\boxed{23}} → 23
        """
        result = self._extract_single_boxed(solution)
        if result is None:
            return None
        
        # 递归提取最内层
        while '\\boxed{' in result:
            inner = self._extract_single_boxed(result)
            if inner is None:
                break
            result = inner
        
        # 清理
        result = result.strip()
        
        # 移除首尾的 $ 符号
        if result.startswith('$') and result.endswith('$'):
            result = result[1:-1].strip()
        
        return result


class DirectAnswerExtractor(AnswerExtractor):
    """
    直接答案提取器 - 不做任何提取，返回 None
    
    适用于: 答案直接存储在 ground_truth 字段的数据集（如 GSM8K-Aug, LILA）
    在 validity.py 中，如果 extract 返回 None，会直接使用 ground_truth
    """
    
    def extract(self, solution: str) -> Optional[str]:
        """不提取答案，返回 None"""
        return None
