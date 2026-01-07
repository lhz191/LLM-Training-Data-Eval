#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xLAM-60k 数据集执行器

包含 xLAM 相关的所有检查逻辑：

1. XLAMFormatChecker - 格式检查
   - 基本结构检查（query, tools, api_calls）
   - 工具定义检查（name, description, parameters）
   - 参数标注一致性检查
   - API 调用检查

2. XLAMExecutabilityChecker - 可执行性检查
   - API 存在性检查 - 调用的 API 是否在工具列表中
   - Required Parameter Check - 必需参数是否完整
   - Argument Type Check - 参数值类型是否与声明类型匹配
"""
from typing import List, Dict, Any, Tuple, Optional

from data_types import APIAgentSample, ToolDefinition, APICall, Parameter
from api_executor import (
    FormatChecker, ExecutabilityChecker,
    register_format_checker, register_executability_checker
)


# =============================================================================
# xLAM 格式检查器
# =============================================================================

class XLAMFormatChecker(FormatChecker):
    """
    xLAM-60k 数据集格式检查器
    
    检查项（基于统一接口）：
    1. 基本结构检查
       - query 是否存在且非空
       - tools 是否存在
       - api_calls 是否存在
       
    2. 工具定义检查
       - 每个 tool 有 name、description
       - parameters 存在
       
    3. 参数标注一致性检查
       - default 在 description 中提及但 default 字段缺失
       - default 值类型与声明类型不匹配
       
    4. API 调用检查
       - 每个调用有 name、arguments
       - 调用的 API 在工具列表中
       - 必填参数都提供了
    """
    
    def check(self, sample: APIAgentSample) -> Tuple[List[str], List[str]]:
        """检查 xLAM 样本的格式正确性"""
        errors = []
        warnings = []
        
        # 1. 基本结构检查
        if not sample.query or not sample.query.strip():
            errors.append("Sample missing or empty 'query'")
        
        if not sample.tools:
            errors.append("Sample has no tools defined")
        
        if not sample.api_calls:
            errors.append("Sample has no API calls (answers)")
        
        # 2. 工具定义检查 + 3. 参数标注一致性检查
        for i, tool in enumerate(sample.tools):
            tool_errors, tool_warnings = self._check_tool_definition(tool, i)
            errors.extend(tool_errors)
            warnings.extend(tool_warnings)
        
        # 4. API 调用检查
        tool_names = [t.name for t in sample.tools]
        tool_map = {t.name: t for t in sample.tools}
        
        for i, call in enumerate(sample.api_calls):
            call_errors, call_warnings = self._check_api_call(call, i, tool_names, tool_map)
            errors.extend(call_errors)
            warnings.extend(call_warnings)
        
        return errors, warnings
    
    def _check_tool_definition(self, tool: ToolDefinition, idx: int) -> Tuple[List[str], List[str]]:
        """检查工具定义 + 参数标注一致性"""
        errors = []
        warnings = []
        
        if not tool.name:
            errors.append(f"Tool {idx}: missing 'name'")
        
        if not tool.description:
            errors.append(f"Tool {idx} ({tool.name}): missing 'description'")
        
        if tool.parameters is None:
            errors.append(f"Tool {idx} ({tool.name}): missing 'parameters'")
        
        # 参数标注一致性检查
        for param in (tool.parameters or []):
            param_errors = self._check_param_annotation(tool.name, idx, param)
            errors.extend(param_errors)
        
        return errors, warnings
    
    def _check_param_annotation(self, tool_name: str, tool_idx: int, param: Parameter) -> List[str]:
        """检查参数标注一致性"""
        errors = []
        description = param.description or ''
        has_default_field = param.default is not None
        default_in_desc = 'default' in description.lower() or 'defaults to' in description.lower()
        
        # Check 1: default mentioned in description but not in 'default' field
        if default_in_desc and not has_default_field:
            errors.append(
                f"Tool {tool_idx} ({tool_name}) param '{param.name}': "
                f"ANNOTATION ISSUE - default mentioned in description but 'default' field missing"
            )
        
        # Check 2: default value type doesn't match declared type
        if has_default_field and param.type:
            base_type = param.type.split(',')[0].strip().lower()
            default_val = param.default
            type_mismatch = None
            
            if base_type == 'int':
                if isinstance(default_val, float) and not default_val.is_integer():
                    type_mismatch = f"type is 'int' but default is float ({default_val})"
                elif isinstance(default_val, str) and default_val != '':
                    try:
                        float(default_val)
                        type_mismatch = f"type is 'int' but default is str ('{default_val}')"
                    except ValueError:
                        pass
            elif base_type == 'str':
                if isinstance(default_val, (int, float)) and default_val != '':
                    type_mismatch = f"type is 'str' but default is {type(default_val).__name__} ({default_val})"
            elif base_type == 'float':
                if isinstance(default_val, str) and default_val != '':
                    try:
                        float(default_val)
                        type_mismatch = f"type is 'float' but default is str ('{default_val}')"
                    except ValueError:
                        pass
            
            if type_mismatch:
                errors.append(
                    f"Tool {tool_idx} ({tool_name}) param '{param.name}': TYPE MISMATCH - {type_mismatch}"
                )
        
        return errors
    
    def _check_api_call(self, call: APICall, idx: int,
                        tool_names: List[str],
                        tool_map: Dict[str, ToolDefinition]) -> Tuple[List[str], List[str]]:
        """检查 API 调用"""
        errors = []
        warnings = []
        
        if not call.name:
            errors.append(f"Answer {idx}: missing 'name'")
            return errors, warnings
        
        if call.arguments is None:
            errors.append(f"Answer {idx} ({call.name}): missing 'arguments'")
        elif not isinstance(call.arguments, dict):
            errors.append(f"Answer {idx} ({call.name}): 'arguments' must be a dict")
        
        # API 必须在工具列表中
        if call.name not in tool_names:
            errors.append(f"Answer {idx}: API '{call.name}' not in available tools")
        
        # 必填参数检查
        if call.name in tool_map:
            tool = tool_map[call.name]
            provided_args = set(call.arguments.keys()) if call.arguments else set()
            
            for param in tool.parameters:
                if not param.optional and param.name not in provided_args:
                    if param.default is None:
                        errors.append(
                            f"Answer {idx} ({call.name}): missing required parameter '{param.name}'"
                        )
        
        return errors, warnings


# =============================================================================
# xLAM 可执行性检查器
# =============================================================================

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

register_format_checker('xlam', XLAMFormatChecker)
register_format_checker('xlam-60k', XLAMFormatChecker)
register_executability_checker('xlam', XLAMExecutabilityChecker)
register_executability_checker('xlam-60k', XLAMExecutabilityChecker)

