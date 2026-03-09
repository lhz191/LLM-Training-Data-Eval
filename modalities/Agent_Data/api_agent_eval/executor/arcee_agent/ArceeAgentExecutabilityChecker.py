#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arcee Agent Data 可执行性检查器（静态检查）

覆盖 5 种子集，统一检查逻辑：
1. API 存在性：调用的 API 在工具列表中
2. 必需参数检查：required 且无 default 的参数必须提供
3. 参数类型检查：传入值的类型与声明类型匹配
"""

import os
import sys
from typing import List, Dict, Any, Tuple, Optional

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import APIAgentSample, ToolDefinition, APICall, Parameter
from api_executor import ExecutabilityChecker


class ArceeAgentExecutabilityChecker(ExecutabilityChecker):
    """
    Arcee Agent Data 可执行性检查器

    检查项：
    1. API 存在性检查
       - 调用的 API 是否在样本工具列表中
       - FinalAction 跳过此检查

    2. 必需参数检查
       - required=True 且 default is None 的参数，调用时必须提供
       - 有 default 值或 optional=True 的参数，未提供不报错

    3. 参数类型检查
       - 传入值的 Python 类型是否与声明类型（str/int/float/bool/list/dict）匹配
    """

    def check(self, sample: APIAgentSample) -> Tuple[List[str], List[str], Dict[str, Any]]:
        errors = []
        warnings = []
        stats = {
            'api_calls_checked': 0,
            'type_mismatches': 0,
            'missing_required': 0,
            'api_not_found': 0,
            'source_dataset': sample.source_dataset or '',
        }

        tool_names = [t.name for t in sample.tools]
        tool_map = {t.name: t for t in sample.tools}

        for i, call in enumerate(sample.api_calls):
            if not call.name:
                continue

            stats['api_calls_checked'] += 1

            # FinalAction 不需要在工具列表中
            if call.name == 'FinalAction':
                continue

            # 1. API 存在性
            if call.name not in tool_names:
                errors.append(f"API Call {i}: '{call.name}' not in available tools")
                stats['api_not_found'] += 1
                continue

            tool = tool_map[call.name]
            args_is_dict = isinstance(call.arguments, dict)
            provided_args = set(call.arguments.keys()) if args_is_dict and call.arguments else set()

            # 2. 必需参数检查
            for param in (tool.parameters or []):
                if param.required and not param.optional and param.name not in provided_args:
                    if param.default is None:
                        errors.append(
                            f"API Call {i} ({call.name}): missing required param "
                            f"'{param.name}'"
                        )
                        stats['missing_required'] += 1

            # 3. 参数类型检查
            if args_is_dict and call.arguments:
                param_map = {p.name: p for p in (tool.parameters or [])}
                for arg_name, arg_value in call.arguments.items():
                    if arg_name in param_map:
                        param = param_map[arg_name]
                        type_err = self._check_argument_type(arg_value, param.type)
                        if type_err:
                            warnings.append(
                                f"API Call {i} ({call.name}) param '{arg_name}': {type_err}"
                            )
                            stats['type_mismatches'] += 1

        return errors, warnings, stats

    def _check_argument_type(self, value: Any, declared_type: Optional[str]) -> Optional[str]:
        if not declared_type:
            return None

        base_type = declared_type.split(',')[0].strip().lower()
        if 'optional' in base_type:
            base_type = base_type.replace('optional', '').strip()

        actual_type = type(value).__name__

        if base_type in ('str', 'string'):
            if not isinstance(value, str):
                return f"expected str, got {actual_type}"
        elif base_type in ('int', 'integer'):
            if not isinstance(value, int) or isinstance(value, bool):
                if isinstance(value, str):
                    try:
                        int(value)
                        return None
                    except ValueError:
                        return f"expected int, got {actual_type} ('{value}')"
                return f"expected int, got {actual_type}"
        elif base_type in ('float', 'number'):
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
