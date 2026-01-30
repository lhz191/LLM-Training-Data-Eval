#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenMath 代码提取器
"""

import os
import sys
import re
from typing import Optional, Tuple, List

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from code_executor import CodeExtractor

class OpenMathCodeExtractor(CodeExtractor):
    """
    OpenMathInstruct-1 代码提取器
    
    格式:
        <llm-code>
        Python 代码
        </llm-code>
        <llm-code-output>
        输出结果
        </llm-code-output>
    
    多代码块说明：
        OpenMath 的 solution 可能包含多个 <llm-code> 块（论文允许最多3个）。
        这些代码块通常是前后依赖的（后续代码块使用前面定义的变量）。
        
        - extract(): 只提取第一个代码块，用于 validity 指标验证
        - extract_all_code(): 提取所有代码块，用于 reasoning_validity 让 LLM 判断完整逻辑
    """
    
    def extract(self, solution: str) -> Optional[str]:
        """提取第一个 <llm-code>...</llm-code> 中的代码"""
        pattern = r'<llm-code>(.*?)</llm-code>'
        match = re.search(pattern, solution, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def extract_all_code(self, solution: str) -> Optional[str]:
        """
        提取所有 <llm-code>...</llm-code> 中的代码，合并返回
        
        用于 reasoning_validity 指标，让 LLM 看到完整的代码逻辑。
        多个代码块之间用分隔符连接。
        """
        pattern = r'<llm-code>(.*?)</llm-code>'
        matches = re.findall(pattern, solution, re.DOTALL)
        if matches:
            if len(matches) == 1:
                return matches[0].strip()
            else:
                # 多个代码块，用分隔符连接
                return '\n\n# --- 代码块分隔 ---\n\n'.join(m.strip() for m in matches)
        return None
    
    def extract_output(self, solution: str) -> Optional[str]:
        """提取第一个 <llm-code-output>...</llm-code-output> 中的预期输出"""
        pattern = r'<llm-code-output>(.*?)</llm-code-output>'
        match = re.search(pattern, solution, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
