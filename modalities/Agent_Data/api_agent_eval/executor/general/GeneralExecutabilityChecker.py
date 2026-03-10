#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用可执行性检查器（静态）

纯粹基于 data_types.py 合同 (APIAgentSample / ToolDefinition / APICall / Parameter)
进行静态可执行性检查，不依赖任何数据集特有逻辑。

检查目标：给定工具定义，API 调用是否"可执行"——
即工具存在、必需参数齐全、参数类型匹配、无未知参数。

适用场景：
- 用户按照 data_types.py 合同自行生成的数据集
- 任何已通过 Loader 转换为 APIAgentSample 的数据集的基线检查
- 作为自定义 DatasetSpecificChecker 的参考实现
"""

import os
import sys
from typing import List, Dict, Any, Tuple, Optional

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import APIAgentSample, ToolDefinition, APICall, Parameter
from api_executor import ExecutabilityChecker


# =============================================================================
# 类型映射表：Parameter.type → Python 类型
# =============================================================================

_TYPE_MAP: Dict[str, tuple] = {
    'str':     (str,),
    'string':  (str,),
    'int':     (int,),
    'integer': (int,),
    'float':   (int, float),
    'number':  (int, float),
    'bool':    (bool,),
    'boolean': (bool,),
    'list':    (list,),
    'array':   (list,),
    'dict':    (dict,),
    'object':  (dict,),
}


class GeneralExecutabilityChecker(ExecutabilityChecker):
    """
    通用可执行性检查器：基于 data_types.py 合同进行静态检查

    逐条遍历 sample.api_calls，对每个 APICall 检查：

    1. 工具存在性 — APICall.name 在 sample.tools 中
       依据: APIAgentSample.tools (List[ToolDefinition])
             ToolDefinition.name (str)

    2. 必需参数完整性 — Parameter.required=True 的参数在 APICall.arguments 中
       依据: ToolDefinition.parameters (List[Parameter])
             Parameter.required (bool)
             Parameter.default (Any) — 有默认值时允许缺失

    3. 参数类型匹配 — APICall.arguments[key] 的 Python 类型与 Parameter.type 一致
       依据: Parameter.type (str) → _TYPE_MAP
       注意: 字符串形式的数字 ("123") 在 int/float 类型时视为 warning 而非 error

    4. 未知参数检测 — APICall.arguments 中出现 ToolDefinition.parameters 里没有的 key
       依据: ToolDefinition.get_all_param_names()
    """

    def check(self, sample: APIAgentSample) -> Tuple[List[str], List[str], Dict[str, Any]]:
        errors: List[str] = []
        warnings: List[str] = []
        stats: Dict[str, Any] = {
            'api_calls_checked': 0,
            'tool_not_found': 0,
            'missing_required': 0,
            'type_mismatches': 0,
            'unknown_params': 0,
        }

        tool_map: Dict[str, ToolDefinition] = {t.name: t for t in sample.tools}

        for i, call in enumerate(sample.api_calls):
            if not call.name:
                continue

            stats['api_calls_checked'] += 1
            prefix = f"api_calls[{i}] '{call.name}'"

            # =================================================================
            # 1. 工具存在性
            # =================================================================
            tool = tool_map.get(call.name)
            if tool is None:
                errors.append(f"{prefix}: tool not found in tools list")
                stats['tool_not_found'] += 1
                continue

            provided_args = set(call.arguments.keys()) if call.arguments else set()
            param_map: Dict[str, Parameter] = {p.name: p for p in tool.parameters}

            # =================================================================
            # 2. 必需参数完整性
            # =================================================================
            for param in tool.parameters:
                if not param.required:
                    continue
                if param.name in provided_args:
                    continue
                if param.default is not None:
                    continue
                errors.append(f"{prefix}: missing required param '{param.name}'")
                stats['missing_required'] += 1

            # =================================================================
            # 3. 参数类型匹配
            # =================================================================
            if call.arguments:
                for arg_name, arg_value in call.arguments.items():
                    if arg_name not in param_map:
                        continue
                    param = param_map[arg_name]
                    type_issue = self._check_type(arg_value, param.type)
                    if type_issue:
                        level, msg = type_issue
                        full_msg = f"{prefix} param '{arg_name}': {msg}"
                        if level == 'error':
                            errors.append(full_msg)
                        else:
                            warnings.append(full_msg)
                        stats['type_mismatches'] += 1

            # =================================================================
            # 4. 未知参数检测
            # =================================================================
            all_param_names = set(tool.get_all_param_names())
            for arg_name in provided_args:
                if arg_name not in all_param_names:
                    warnings.append(f"{prefix}: unknown param '{arg_name}'")
                    stats['unknown_params'] += 1

        return errors, warnings, stats

    @staticmethod
    def _check_type(value: Any, declared_type: Optional[str]) -> Optional[Tuple[str, str]]:
        """
        检查参数值与 Parameter.type 的匹配性

        Returns:
            None 如果匹配；否则 (level, message)
            level = 'error' | 'warning'
        """
        if declared_type is None or declared_type == '':
            return None

        base_type = declared_type.split(',')[0].strip().lower()
        if 'optional' in base_type:
            base_type = base_type.replace('optional', '').strip()

        if value is None:
            return None

        expected_types = _TYPE_MAP.get(base_type)
        if expected_types is None:
            return None

        # bool 是 int 的子类，需要特殊处理
        if base_type in ('int', 'integer', 'float', 'number') and isinstance(value, bool):
            return ('error', f"expected {base_type}, got bool")

        if isinstance(value, expected_types):
            return None

        # 字符串形式的数字：降级为 warning
        if base_type in ('int', 'integer') and isinstance(value, str):
            try:
                int(value)
                return ('warning', f"expected int, got str (parseable: '{value}')")
            except ValueError:
                return ('error', f"expected int, got str (not parseable: '{value}')")

        if base_type in ('float', 'number') and isinstance(value, str):
            try:
                float(value)
                return ('warning', f"expected {base_type}, got str (parseable: '{value}')")
            except ValueError:
                return ('error', f"expected {base_type}, got str (not parseable: '{value}')")

        actual = type(value).__name__
        return ('error', f"expected {base_type}, got {actual}")
