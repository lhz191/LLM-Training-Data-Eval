#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xLAM 可执行性检查器（静态 + LLM Judge）

检查逻辑分三层：
1. 定义层（API Definition）：调用的 API 在工具列表中
2. 参数层（Parameter）：必需参数是否提供、类型是否匹配
3. 调用层（Invocation）：
   - Tool Selection Relevance：选的 API 能不能解决用户需求（LLM Judge）
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Tuple, Optional

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import APIAgentSample, ToolDefinition, APICall, Parameter
from api_executor import ExecutabilityChecker

from .constants import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    TOOL_SELECTION_PROMPT,
)


class XLAMExecutabilityChecker(ExecutabilityChecker):
    """
    xLAM-60k 数据集可执行性检查器

    检查项：
    1. API 存在性检查（定义层）
       - 调用的 API 是否在工具列表中

    2. Required Parameter Check（参数层）
       - 必需参数是否完整

    3. Argument Type Check（参数层）
       - 参数值类型是否与声明类型匹配

    4. Tool Selection Relevance（调用层，LLM Judge）
       - 选的 API 是否能解决用户的 query
       - xLAM 没有 final_answer / response，直接判断 query → tool_calls 的合理性
    """

    def check(self, sample: APIAgentSample) -> Tuple[List[str], List[str], Dict[str, Any]]:
        errors = []
        warnings = []
        stats = {
            'api_calls_checked': 0,
            'type_mismatches': 0,
            'tool_selection_relevance': None,
        }

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

        # 4. Tool Selection（调用层）—— API 调用能否满足用户需求
        if sample.query and sample.api_calls:
            satisfied, reason = self._check_tool_selection_llm(sample)
            stats['tool_selection_relevance'] = {
                'satisfied': satisfied,
                'reason': reason,
            }
            if not satisfied:
                errors.append(f"Tool Selection: {reason}")

        return errors, warnings, stats

    # =====================================================================
    # 参数类型检查
    # =====================================================================

    def _check_argument_type(self, value: Any, declared_type: Optional[str]) -> Optional[str]:
        if not declared_type:
            return None

        base_type = declared_type.split(',')[0].strip().lower()
        if 'optional' in base_type:
            base_type = base_type.replace('optional', '').strip()

        actual_type = type(value).__name__

        if base_type == 'str':
            if not isinstance(value, str):
                return f"expected str, got {actual_type}"
        elif base_type == 'int':
            if not isinstance(value, int) or isinstance(value, bool):
                if isinstance(value, str):
                    try:
                        int(value)
                        return None
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

    # =====================================================================
    # LLM Judge: Tool Selection Relevance
    # =====================================================================

    def _check_tool_selection_llm(
        self,
        sample: APIAgentSample,
        max_retries: int = 3
    ) -> Tuple[bool, str]:
        """
        使用 LLM 判断 API 调用是否能满足用户需求。
        只传被调用的 API 信息（名称 + 描述 + 参数），不传可用工具列表。
        """
        try:
            from openai import OpenAI
        except ImportError:
            return False, "OpenAI library not installed"

        # 构建 tool_map 用于查找 API 描述
        tool_map = {t.name: t for t in sample.tools}

        # 只传被调用的 API 信息
        calls_desc = []
        for i, call in enumerate(sample.api_calls):
            tool = tool_map.get(call.name)
            description = tool.description if tool else "no description"
            args_str = json.dumps(call.arguments, ensure_ascii=False) if call.arguments else "{}"
            calls_desc.append(
                f"{i+1}. {call.name}\n"
                f"   描述: {description}\n"
                f"   参数: {args_str}"
            )
        calls_text = "\n\n".join(calls_desc)

        prompt = TOOL_SELECTION_PROMPT.format(
            query=sample.query,
            api_calls=calls_text
        )

        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )

                content = response.choices[0].message.content.strip()

                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                result = json.loads(content)
                satisfied = result.get('satisfied', False)
                reason = result.get('reason', '')

                return satisfied, reason

            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2)

        return False, "LLM call failed"


# =============================================================================
# 注册到全局注册表
# =============================================================================
