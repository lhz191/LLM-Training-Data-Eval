#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenMath 格式检查器
"""

import os
import sys
from typing import List, Tuple

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from code_executor import FormatChecker

class OpenMathFormatChecker(FormatChecker):
    """
    OpenMathInstruct-1 数据集格式检查器
    
    基于论文 Section 2.1 和 Section 2.3 的要求：
    
    必需字段检查：
    1. question: 问题文本（不能为空）
    2. solution: 解答文本（不能为空）
    3. ground_truth: 标准答案（不能为空）
    
    Solution 格式检查（论文 Section 2.3 Post-processing）：
    1. <llm-code> 和 </llm-code> 标签必须配对
    2. <llm-code-output> 和 </llm-code-output> 标签必须配对
    3. 不能有多个 \\boxed{} 块（论文明确要求移除，嵌套的算一个）
    """
    
    def _count_top_level_boxed(self, text: str) -> int:
        """
        统计顶层（非嵌套）的 \\boxed{} 数量
        
        例如：
        - "\\boxed{30}" -> 1
        - "\\boxed{\\boxed{30}}" -> 1 (嵌套算一个)
        - "\\boxed{30} and \\boxed{40}" -> 2 (两个独立的)
        """
        count = 0
        i = 0
        boxed_marker = '\\boxed{'
        
        while i < len(text):
            # 找下一个 \boxed{
            pos = text.find(boxed_marker, i)
            if pos == -1:
                break
            
            # 找到一个顶层 boxed，计数+1
            count += 1
            
            # 跳过这个 boxed 的内容（包括嵌套的）
            # 从 { 开始匹配括号
            brace_start = pos + len(boxed_marker) - 1  # 指向 {
            depth = 1
            j = brace_start + 1
            
            while j < len(text) and depth > 0:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            
            # 跳到这个 boxed 结束后继续搜索
            i = j
        
        return count
    
    def check(self, sample) -> Tuple[List[str], List[str]]:
        """检查 OpenMath 样本的格式正确性"""
        errors = []
        warnings = []
        
        # === 1. 必需字段检查 ===
        if not sample.question:
            errors.append("Missing or empty 'question' field")
        
        if not sample.solution:
            errors.append("Missing or empty 'solution' field")
        
        if sample.ground_truth is None:
            errors.append("Missing 'ground_truth' field")
        elif isinstance(sample.ground_truth, str) and not sample.ground_truth.strip():
            errors.append("Empty 'ground_truth' field")
        elif isinstance(sample.ground_truth, list) and len(sample.ground_truth) == 0:
            errors.append("Empty 'ground_truth' list")
        
        # === 2. solution 格式检查 ===
        if sample.solution:
            solution = sample.solution
            if isinstance(solution, list):
                solution = "\n".join(solution)
            
            # 检查代码标签配对（论文 Section 2.3：移除有 <llm-code> 但没有 </llm-code> 的）
            code_open = solution.count('<llm-code>')
            code_close = solution.count('</llm-code>')
            if code_open != code_close:
                errors.append(f"Mismatched <llm-code> tags: {code_open} open, {code_close} close")
            
            # 检查代码输出标签配对
            output_open = solution.count('<llm-code-output>')
            output_close = solution.count('</llm-code-output>')
            if output_open != output_close:
                errors.append(f"Mismatched <llm-code-output> tags: {output_open} open, {output_close} close")
            
            # 检查多个 \boxed{}（论文 Section 2.3：移除有多个 \boxed{} 的）
            # 注意：嵌套的 \boxed{} 算一个，只统计顶层的
            boxed_count = self._count_top_level_boxed(solution)
            if boxed_count > 1:
                warnings.append(f"Multiple \\boxed{{}} blocks found: {boxed_count}")
        
        return errors, warnings
