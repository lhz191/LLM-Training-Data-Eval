#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arcee Agent Data 可执行性检查器（静态 + LLM Judge）

覆盖 5 种子集，检查逻辑分三层：
1. 定义层（API Definition）：调用的 API 在工具列表中
2. 参数层（Parameter）：必需参数是否提供、类型是否匹配
3. 调用层（Invocation）：
   - Derivability：final_answer 是否能从 API 响应推导（需要 response）
   - Relevance：final_answer 是否回答了用户 query
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
    DERIVABILITY_PROMPT,
    RELEVANCE_PROMPT,
)


class ArceeAgentExecutabilityChecker(ExecutabilityChecker):
    """
    Arcee Agent Data 可执行性检查器

    检查项：
    1. API 存在性检查（定义层）
       - 调用的 API 是否在样本工具列表中
       - FinalAction 跳过此检查

    2. 必需参数检查（参数层）
       - required=True 且 default is None 的参数，调用时必须提供

    3. 参数类型检查（参数层）
       - 传入值的 Python 类型是否与声明类型匹配

    4. Train Derivability（调用层，LLM Judge）
       - final_answer 是否能从 API 响应中推导
       - 仅对有 response 的子集生效（toolbench_react, toolbench_tflan）

    5. Query-Answer Relevance（调用层，LLM Judge）
       - final_answer 是否回答了用户的 query
       - 对所有有 final_answer 的子集生效
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
            'train_derivability': None,
            'query_relevance': None,
        }

        tool_names = [t.name for t in sample.tools]
        tool_map = {t.name: t for t in sample.tools}

        # ---- 1 & 2 & 3: 静态检查（定义层 + 参数层）----
        for i, call in enumerate(sample.api_calls):
            if not call.name:
                continue

            stats['api_calls_checked'] += 1

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

        # ---- 4. Train Derivability（调用层）----
        if sample.final_answer:
            all_responses = []
            for call in sample.api_calls:
                if call.name not in ('FinalAction', 'Finish') and call.response:
                    all_responses.append(f"[{call.name}]: {str(call.response)}")

            if all_responses:
                all_responses_text = "\n".join(all_responses)
                derivable, reason = self._check_derivability_llm(
                    sample.final_answer, all_responses_text
                )
                stats['train_derivability'] = {
                    'derivable': derivable,
                    'reason': reason,
                }
                if not derivable:
                    errors.append(f"Train Derivability: {reason}")

        # ---- 5. Query-Answer Relevance（调用层）----
        if sample.final_answer and sample.query:
            relevant, reason = self._check_relevance_llm(
                sample.query, sample.final_answer
            )
            stats['query_relevance'] = {
                'relevant': relevant,
                'reason': reason,
            }
            if not relevant:
                errors.append(f"Query-Answer Relevance: {reason}")

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

    # =====================================================================
    # LLM Judge: Derivability
    # =====================================================================

    def _check_derivability_llm(
        self,
        final_answer: str,
        api_responses: str,
        max_retries: int = 3
    ) -> Tuple[bool, str]:
        """
        使用 LLM 判断 final_answer 是否能从 API 响应中推导。
        仅在子集有 call.response 时才会被调用（toolbench_react, toolbench_tflan）。
        """
        try:
            from openai import OpenAI
        except ImportError:
            return False, "OpenAI library not installed"

        prompt = DERIVABILITY_PROMPT.format(
            api_responses=api_responses,
            final_answer=final_answer
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
                derivable = result.get('derivable', False)
                reason = result.get('reason', '')

                return derivable, reason

            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2)

        return False, "LLM call failed"

    # =====================================================================
    # LLM Judge: Relevance
    # =====================================================================

    def _check_relevance_llm(
        self,
        query: str,
        final_answer: str,
        max_retries: int = 3
    ) -> Tuple[bool, str]:
        """
        使用 LLM 判断 final_answer 是否回答了用户 query。
        对所有有 final_answer 的子集生效。
        """
        try:
            from openai import OpenAI
        except ImportError:
            return False, "OpenAI library not installed"

        prompt = RELEVANCE_PROMPT.format(
            query=query,
            final_answer=final_answer
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
                relevant = result.get('relevant', False)
                reason = result.get('reason', '')

                return relevant, reason

            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2)

        return False, "LLM call failed"
