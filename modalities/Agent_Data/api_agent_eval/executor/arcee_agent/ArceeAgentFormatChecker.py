#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arcee Agent Data 格式检查器

支持 5 种子集格式，根据 source_dataset 自动路由：
1. glaive-function-calling-v2-extended
2. salesforce_sharegpt
3. toolbench_instruct_j1s1_3k_unfiltered
4. toolbench_react_10p_unfiltered
5. toolbench_tflan_cot_30p_unfiltered

注意：三个 toolbench_* 子集内部 key 命名极其多样化（大小写、别名均不同），
是数据集构造时有意做的多样化，因此子集检查只做结构性验证，不检查具体 key 名。
"""

import os
import sys
import re
import json
from typing import List, Dict, Any, Tuple, Optional

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import APIAgentSample, ToolDefinition, APICall, Parameter
from api_executor import FormatChecker


class ArceeAgentFormatChecker(FormatChecker):
    """
    Arcee Agent Data 格式检查器

    通用检查项（所有子集）：
    1. 基本结构：query / tools / api_calls 存在性
    2. 工具定义：name / description / parameters 完整性
    3. 参数一致性：required 与 optional 互补
    4. API 调用：name 存在、调用的 API 在工具列表中、必填参数已提供

    子集特有检查项（仅做结构性验证，不强制 key 名）：
    - glaive: system prompt 和 function definition 结构
    - salesforce: <tools>/<tool_call> 标签结构
    - toolbench_*: assistant 返回可解析 JSON / 对话结构完整性
    """

    def check(self, sample: APIAgentSample) -> Tuple[List[str], List[str]]:
        errors = []
        warnings = []

        # === 通用检查 ===
        e, w = self._check_common(sample)
        errors.extend(e)
        warnings.extend(w)

        # === 子集特有检查 ===
        dataset = sample.source_dataset or ''
        if dataset == 'glaive-function-calling-v2-extended':
            e, w = self._check_glaive(sample)
        elif dataset == 'salesforce_sharegpt':
            e, w = self._check_salesforce(sample)
        elif dataset == 'toolbench_instruct_j1s1_3k_unfiltered':
            e, w = self._check_toolbench_instruct(sample)
        elif dataset == 'toolbench_react_10p_unfiltered':
            e, w = self._check_toolbench_react(sample)
        elif dataset == 'toolbench_tflan_cot_30p_unfiltered':
            e, w = self._check_toolbench_tflan(sample)
        else:
            e, w = [], [f"Unknown sub-dataset: '{dataset}', skipping format-specific checks"]
        errors.extend(e)
        warnings.extend(w)

        return errors, warnings

    # =========================================================================
    # 通用检查
    # =========================================================================

    def _check_common(self, sample: APIAgentSample) -> Tuple[List[str], List[str]]:
        errors = []
        warnings = []

        if not sample.query or not sample.query.strip():
            errors.append("Missing or empty 'query'")

        if not sample.tools:
            errors.append("No tools defined")

        if not sample.api_calls:
            warnings.append("No API calls found")

        for i, tool in enumerate(sample.tools):
            te, tw = self._check_tool_definition(tool, i)
            errors.extend(te)
            warnings.extend(tw)

        tool_names = [t.name for t in sample.tools]
        tool_map = {t.name: t for t in sample.tools}
        for i, call in enumerate(sample.api_calls):
            ce, cw = self._check_api_call(call, i, tool_names, tool_map)
            errors.extend(ce)
            warnings.extend(cw)

        return errors, warnings

    def _check_tool_definition(self, tool: ToolDefinition, idx: int) -> Tuple[List[str], List[str]]:
        errors = []
        warnings = []

        if not tool.name:
            errors.append(f"Tool {idx}: missing 'name'")

        if not tool.description:
            warnings.append(f"Tool {idx} ({tool.name}): missing 'description'")

        for param in (tool.parameters or []):
            if not param.name:
                errors.append(f"Tool {idx} ({tool.name}): parameter missing 'name'")
            if param.required == param.optional:
                if param.required and param.optional:
                    errors.append(
                        f"Tool {idx} ({tool.name}) param '{param.name}': "
                        f"both required and optional are True"
                    )
                elif not param.required and not param.optional:
                    errors.append(
                        f"Tool {idx} ({tool.name}) param '{param.name}': "
                        f"both required and optional are False"
                    )

        return errors, warnings

    def _check_api_call(self, call: APICall, idx: int,
                        tool_names: List[str],
                        tool_map: Dict[str, ToolDefinition]) -> Tuple[List[str], List[str]]:
        errors = []
        warnings = []

        if not call.name:
            errors.append(f"API Call {idx}: missing 'name'")
            return errors, warnings

        args_is_dict = isinstance(call.arguments, dict)
        if call.arguments is not None and not args_is_dict:
            errors.append(f"API Call {idx} ({call.name}): 'arguments' is not a dict (got {type(call.arguments).__name__})")

        if call.name != 'FinalAction' and call.name not in tool_names:
            errors.append(f"API Call {idx}: '{call.name}' not in available tools")

        if call.name in tool_map and args_is_dict:
            tool = tool_map[call.name]
            provided = set(call.arguments.keys()) if call.arguments else set()
            for param in (tool.parameters or []):
                if param.required and not param.optional and param.name not in provided:
                    if param.default is None:
                        errors.append(
                            f"API Call {idx} ({call.name}): missing required param '{param.name}'"
                        )

        return errors, warnings

    # =========================================================================
    # glaive-function-calling-v2-extended
    # =========================================================================

    def _check_glaive(self, sample: APIAgentSample) -> Tuple[List[str], List[str]]:
        """
        glaive 格式检查：
        - system prompt 存在且含 function 定义
        - 数据格式统一（OpenAI function calling JSON），可做结构检查
        """
        errors = []
        warnings = []
        metadata = sample.metadata or {}
        system_prompt = metadata.get('system_prompt', '')

        if not system_prompt:
            errors.append("[glaive] Missing system prompt")
            return errors, warnings

        if 'no access to external functions' in system_prompt.lower():
            errors.append("[glaive] System declares no access to external functions")
            return errors, warnings

        if '[' not in system_prompt:
            warnings.append("[glaive] System prompt missing function definition JSON array")

        return errors, warnings

    # =========================================================================
    # salesforce_sharegpt
    # =========================================================================

    def _check_salesforce(self, sample: APIAgentSample) -> Tuple[List[str], List[str]]:
        """
        salesforce 格式检查：
        - system prompt 含 <tools></tools> 标签
        - 数据格式统一，可做结构检查
        """
        errors = []
        warnings = []
        metadata = sample.metadata or {}
        system_prompt = metadata.get('system_prompt', '')

        if '<tools>' not in system_prompt:
            errors.append("[salesforce] System prompt missing <tools> tag")

        if '</tools>' not in system_prompt:
            errors.append("[salesforce] System prompt missing </tools> closing tag")

        return errors, warnings

    # =========================================================================
    # toolbench_instruct_j1s1_3k_unfiltered
    # =========================================================================

    def _check_toolbench_instruct(self, sample: APIAgentSample) -> Tuple[List[str], List[str]]:
        """
        toolbench_instruct 格式检查：
        - 单轮对话，assistant 返回 JSON
        - key 名高度多样化（NAME/Name/name/Tool/Action 等均有），不检查具体 key
        - 只检查 assistant 回复是否为合法 JSON
        """
        errors = []
        warnings = []
        metadata = sample.metadata or {}

        convs = metadata.get('raw_conversations', [])
        assistant_texts = [c.get('value', '') for c in convs
                          if c.get('from', '').lower() in ('assistant', 'gpt')]

        for i, text in enumerate(assistant_texts):
            text_stripped = text.strip()
            if text_stripped.startswith('{'):
                try:
                    json.loads(text_stripped)
                except json.JSONDecodeError:
                    warnings.append(f"[tb_instruct] Assistant msg {i}: invalid JSON")

        return errors, warnings

    # =========================================================================
    # toolbench_react_10p_unfiltered
    # =========================================================================

    def _check_toolbench_react(self, sample: APIAgentSample) -> Tuple[List[str], List[str]]:
        """
        toolbench_react 格式检查：
        - 多轮对话，assistant 返回 JSON
        - key 名高度多样化（action/Action/Command/API/Tool 等均有），不检查具体 key
        - 只检查 JSON 可解析性和 FinalAction 结构
        """
        errors = []
        warnings = []

        # FinalAction 结尾检查（如果有 FinalAction）
        if sample.api_calls:
            last_call = sample.api_calls[-1]
            if last_call.name == 'FinalAction':
                args = last_call.arguments if isinstance(last_call.arguments, dict) else {}
                rt = args.get('return_type', '')
                if rt not in ('give_answer', 'give_up_and_restart'):
                    warnings.append(
                        f"[tb_react] FinalAction return_type='{rt}', "
                        f"expected 'give_answer' or 'give_up_and_restart'"
                    )
                if rt == 'give_answer' and 'final_answer' not in args:
                    warnings.append("[tb_react] FinalAction give_answer missing 'final_answer'")

        # assistant JSON 可解析性
        metadata = sample.metadata or {}
        convs = metadata.get('raw_conversations', [])
        for i, c in enumerate(convs):
            if c.get('from', '').lower() not in ('assistant', 'gpt'):
                continue
            text = c.get('value', '').strip()
            if text.startswith('{'):
                try:
                    json.loads(text)
                except json.JSONDecodeError:
                    warnings.append(f"[tb_react] Conv {i}: invalid JSON in assistant response")

        return errors, warnings

    # =========================================================================
    # toolbench_tflan_cot_30p_unfiltered
    # =========================================================================

    def _check_toolbench_tflan(self, sample: APIAgentSample) -> Tuple[List[str], List[str]]:
        """
        toolbench_tflan_cot 格式检查：
        - 多轮对话，assistant 回复可能含 ```python 代码块，也可能是纯自然语言
        - 不强制要求 ```python 块（实测约 2/3 的非末尾 msg 没有代码块）
        - 只检查对话结构完整性
        """
        errors = []
        warnings = []
        metadata = sample.metadata or {}
        convs = metadata.get('raw_conversations', [])

        assistant_msgs = [c for c in convs
                          if c.get('from', '').lower() in ('assistant', 'gpt')]

        if not assistant_msgs:
            errors.append("[tb_tflan] No assistant messages found")

        return errors, warnings
