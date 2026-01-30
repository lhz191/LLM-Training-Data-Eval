#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xLAM 可执行性检查器
"""

import os
import sys
from typing import List, Dict, Any, Tuple, Optional

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import APIAgentSample, ToolDefinition, APICall, Parameter
from api_executor import ExecutabilityChecker

class XLAMExecutabilityChecker(ExecutabilityChecker):
    """
    xLAM-60k 数据集可执行性检查器
    
    复用自 evaluate_xlam_basic.py.evaluate_executability
    
    检查项：
    1. API 存在性检查
       - 调用的 API 是否在工具列表中
       
    2. Required Parameter Check
       - 必需参数是否完整
       
    3. Argument Type Check
       - 参数值类型是否与声明类型匹配
    """
    
    def check(self, sample: APIAgentSample) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """检查 xLAM 样本的可执行性"""
        errors = []
        warnings = []
        stats = {
            'api_calls_checked': 0,
            'type_mismatches': 0,
        }
        
        # 构建工具名映射
        tool_names = [t.name for t in sample.tools]
        tool_map = {t.name: t for t in sample.tools}
        
        for i, call in enumerate(sample.api_calls):
            if not call.name:
                continue
            
            stats['api_calls_checked'] += 1
            
            # 1. API 存在性检查
            if call.name not in tool_names:
                errors.append(f"Answer {i}: API '{call.name}' not in available tools")
                continue
            
            # 2. Required Parameter Check
            if call.name in tool_map:
                tool = tool_map[call.name]
                provided_args = set(call.arguments.keys()) if call.arguments else set()
                
                for param in (tool.parameters or []):
                    if not param.optional and param.name not in provided_args:
                        if param.default is None:
                            errors.append(
                                f"Answer {i} ({call.name}): missing required parameter '{param.name}'"
                            )
                
                # 3. Argument Type Check
                if call.arguments:
                    param_map = {p.name: p for p in (tool.parameters or [])}
                    for arg_name, arg_value in call.arguments.items():
                        if arg_name in param_map:
                            param = param_map[arg_name]
                            type_error = self._check_argument_type(arg_value, param.type)
                            if type_error:
                                warnings.append(
                                    f"Answer {i} ({call.name}) param '{arg_name}': {type_error}"
                                )
                                stats['type_mismatches'] += 1
        
        return errors, warnings, stats
    
    def _check_argument_type(self, value: Any, declared_type: Optional[str]) -> Optional[str]:
        """
        检查参数值类型是否与声明类型匹配
        
        Args:
            value: 参数值
            declared_type: 声明的类型字符串（如 "str", "int", "float"）
            
        Returns:
            错误信息，如果匹配则返回 None
        """
        if not declared_type:
            return None
        
        # 提取基础类型
        base_type = declared_type.split(',')[0].strip().lower()
        
        # 移除 "optional" 标记
        if 'optional' in base_type:
            base_type = base_type.replace('optional', '').strip()
        
        actual_type = type(value).__name__
        
        # 检查类型匹配
        if base_type == 'str':
            if not isinstance(value, str):
                return f"expected str, got {actual_type}"
        elif base_type == 'int':
            if not isinstance(value, int) or isinstance(value, bool):
                # 允许数字字符串
                if isinstance(value, str):
                    try:
                        int(value)
                        return None  # 数字字符串可以接受
                    except ValueError:
                        return f"expected int, got {actual_type} ('{value}')"
                return f"expected int, got {actual_type}"
        elif base_type == 'float':
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return f"expected float, got {actual_type}"
        elif base_type in ('bool', 'boolean'):
            if not isinstance(value, bool):
                return f"expected bool, got {actual_type}"
        elif base_type in ('list', 'array'):
            if not isinstance(value, list):
                return f"expected list, got {actual_type}"
        elif base_type in ('dict', 'object'):
            if not isinstance(value, dict):
                return f"expected dict, got {actual_type}"
        
        return None


# =============================================================================
# 注册到全局注册表
# =============================================================================


