#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用格式检查器

纯粹基于 data_types.py 合同 (APIAgentSample / ToolDefinition / APICall / Parameter)
进行格式检查，不依赖任何数据集特有逻辑。

适用场景：
- 用户按照 data_types.py 合同自行生成的数据集
- 任何已通过 Loader 转换为 APIAgentSample 的数据集的基线检查
- 作为自定义 DatasetSpecificChecker 的参考实现
"""

import os
import sys
from typing import List, Dict, Any, Tuple, Optional, Set

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import APIAgentSample, ToolDefinition, APICall, Parameter
from api_executor import FormatChecker


class GeneralFormatChecker(FormatChecker):
    """
    通用格式检查器：基于 data_types.py 合同本身进行检查

    检查项（全部来源于 dataclass 字段定义）：

    1. APIAgentSample 层
       - query (str, 必需): 非空
       - tools (List[ToolDefinition], 必需): 非空列表
       - api_calls (List[APICall], 必需): 非空列表
       - final_answer (Optional[str]): 如存在则非空字符串

    2. ToolDefinition 层
       - name (str, 必需): 非空
       - description (str): 建议非空 (warning)
       - parameters (List[Parameter]): 列表元素合法

    3. Parameter 层
       - name (str, 必需): 非空
       - type (str, 必需): 非空
       - required/optional 互补性: required=True 时 optional 应为 False

    4. APICall 层
       - name (str, 必需): 非空
       - name 存在于 tools 中
       - arguments (Dict): 是 dict 类型

    5. 全局一致性
       - 工具名无重复
       - 参数名在同一工具内无重复
    """

    def check(self, sample: APIAgentSample) -> Tuple[List[str], List[str]]:
        errors = []
        warnings = []

        # =====================================================================
        # 1. APIAgentSample 层: query, tools, api_calls, final_answer
        # =====================================================================
        if not sample.query or not sample.query.strip():
            errors.append("query is empty or missing")

        if not sample.tools:
            errors.append("tools list is empty or missing")

        if not sample.api_calls:
            errors.append("api_calls list is empty or missing")

        if sample.final_answer is not None and not sample.final_answer.strip():
            warnings.append("final_answer is present but empty string")

        # =====================================================================
        # 2. ToolDefinition 层 + 3. Parameter 层
        # =====================================================================
        tool_names_seen: Set[str] = set()

        for i, tool in enumerate(sample.tools):
            prefix = f"tools[{i}]"

            # ToolDefinition.name (str, 必需)
            if not tool.name:
                errors.append(f"{prefix}: name is empty")
            else:
                if tool.name in tool_names_seen:
                    warnings.append(f"{prefix}: duplicate tool name '{tool.name}'")
                tool_names_seen.add(tool.name)

            # ToolDefinition.description (str, 建议非空)
            if not tool.description:
                warnings.append(f"{prefix} '{tool.name}': description is empty")

            # ToolDefinition.parameters (List[Parameter])
            param_names_seen: Set[str] = set()
            for j, param in enumerate(tool.parameters):
                p_prefix = f"{prefix}.parameters[{j}]"

                # Parameter.name (str, 必需)
                if not param.name:
                    errors.append(f"{p_prefix}: name is empty")
                else:
                    if param.name in param_names_seen:
                        warnings.append(
                            f"{p_prefix}: duplicate param name '{param.name}' in tool '{tool.name}'"
                        )
                    param_names_seen.add(param.name)

                # Parameter.type (str, 必需)
                if not param.type:
                    warnings.append(f"{p_prefix} '{param.name}': type is empty")

                # Parameter.required / optional 互补性
                if param.required and param.optional:
                    warnings.append(
                        f"{p_prefix} '{param.name}': required=True AND optional=True "
                        f"(should be mutually exclusive)"
                    )
                if not param.required and not param.optional:
                    warnings.append(
                        f"{p_prefix} '{param.name}': required=False AND optional=False "
                        f"(at least one should be True)"
                    )

        # =====================================================================
        # 4. APICall 层
        # =====================================================================
        for k, call in enumerate(sample.api_calls):
            prefix = f"api_calls[{k}]"

            # APICall.name (str, 必需)
            if not call.name:
                errors.append(f"{prefix}: name is empty")
                continue

            # APICall.name 存在于 tools 中
            if tool_names_seen and call.name not in tool_names_seen:
                errors.append(
                    f"{prefix}: API '{call.name}' not found in tools "
                    f"(available: {sorted(tool_names_seen)})"
                )

            # APICall.arguments 类型检查
            if call.arguments is not None and not isinstance(call.arguments, dict):
                errors.append(
                    f"{prefix} '{call.name}': arguments should be dict, "
                    f"got {type(call.arguments).__name__}"
                )

        return errors, warnings
