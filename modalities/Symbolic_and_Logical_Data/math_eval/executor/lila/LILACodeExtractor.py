#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LILA 代码提取器
"""

import os
import sys
from typing import Optional

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from code_executor import CodeExtractor

class LILACodeExtractor(CodeExtractor):
    """
    LILA 数据集代码提取器
    
    LILA 的 solution 本身就是 Python 代码（不需要从 markdown 中提取）
    """
    
    def extract(self, solution: str) -> Optional[str]:
        """
        LILA 的 solution 就是代码，直接返回
        """
        if solution and solution.strip():
            return solution.strip()
        return None
    
    def extract_all_code(self, solution: str) -> Optional[str]:
        """
        LILA 的 solution 就是代码，直接调用 extract()
        
        与 OpenMathCodeExtractor.extract_all_code() 保持接口一致
        """
        return self.extract(solution)
    
    def extract_output(self, solution: str) -> Optional[str]:
        """
        LILA 没有嵌入的预期输出，返回 None
        """
        return None


