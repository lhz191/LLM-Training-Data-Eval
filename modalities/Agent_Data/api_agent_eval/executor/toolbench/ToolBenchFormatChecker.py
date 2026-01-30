#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToolBench 格式检查器
"""

import os
import sys
import re
import json
from typing import List, Dict, Any, Tuple, Optional

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import APIAgentSample, ToolDefinition, APICall, Parameter
from api_executor import FormatChecker

class ToolBenchFormatChecker(FormatChecker):
    """
    ToolBench 数据集格式检查器
    
    检查项（基于统一接口 + 需要时从 metadata 读取原始数据）：
    
    1. 必需字段检查（样本级别）
       - id 字段是否存在（从 metadata['original_id'] 检查）
       - conversations 字段是否存在（从 metadata['roles_in_conversations'] 检查）
    
    2. id 格式检查（样本级别，从 metadata 读取）
       - 必须以 Step N: 开头
       - 能否正确解析 Step 数字
    
    3. conversations 结构检查（样本级别，从 metadata 读取）
       - 必须包含 system
       - 必须包含 user
       - 必须包含 assistant
    
    4. API 定义格式检查（统一接口，system prompt 中）
       - 每个 API 必须有 name
       - 每个 API 必须有 description
       - 每个 API 必须有 parameters 对象
       - Finish 函数额外检查：return_type 必须在 required 中
    
    5. Assistant Action 格式检查（从 api_call.metadata['raw_assistant_text'] 读取）
       - 必须有 Thought
       - 必须有 Action
       - 必须有 Action Input
       
       5.1 Thought 长度检查（论文约束：最多5句话）
       5.2 Action Input JSON 格式检查
       5.3 Finish函数语义检查
    
    6. API 调用检查（统一接口）
       - 每个调用有 name、arguments
       - 调用的 API 在工具列表中（Finish 除外）
       - 必填参数都提供了
    """
    
    # 论文约束：Thought 最多 5 句话
    MAX_THOUGHT_SENTENCES = 5
    
    def check(self, sample: APIAgentSample) -> Tuple[List[str], List[str]]:
        """检查 ToolBench 样本的格式正确性"""
        errors = []
        warnings = []
        
        metadata = sample.metadata or {}
        
        # === 1. 必需字段检查（从 metadata 读取原始数据） ===
        # 检查 id 字段
        original_id = metadata.get('original_id')
        if original_id is None or original_id == '':
            errors.append("Missing 'id' field")
        else:
            # 检查 id 格式：必须以 Step N: 开头
            if not original_id.startswith('Step '):
                errors.append(f"Invalid id format: should start with 'Step N:'")
            else:
                # 尝试解析 Step 数字
                if not re.match(r'Step \d+:', original_id):
                    errors.append(f"Cannot parse Step number from id")
        
        # 检查 conversations 字段
        roles = metadata.get('roles_in_conversations')
        if roles is None:
            errors.append("Missing 'conversations' field")
        elif len(roles) == 0:
            errors.append("Empty conversations")
        else:
            # === 2. conversations 结构检查 ===
            if 'system' not in roles:
                errors.append("Missing 'system' in conversations")
            if 'user' not in roles:
                errors.append("Missing 'user' in conversations")
            if 'assistant' not in roles:
                errors.append("Missing 'assistant' in conversations")
        
        # === 3. 基本结构检查（统一接口） ===
        if not sample.query or not sample.query.strip():
            errors.append("Sample missing or empty 'query'")
        
        if not sample.tools:
            errors.append("Sample has no tools defined")
        
        if not sample.api_calls:
            errors.append("Sample has no API calls")
        
        # === 4. 工具定义检查（统一接口） ===
        for i, tool in enumerate(sample.tools):
            tool_errors, tool_warnings = self._check_tool_definition(tool, i)
            errors.extend(tool_errors)
            warnings.extend(tool_warnings)
        
        # === 5. Assistant Action 格式检查（从 metadata 读取原始文本） ===
        for i, call in enumerate(sample.api_calls):
            action_format_errors = self._check_action_format(call, i)
            errors.extend(action_format_errors)
        
        # === 6. Thought 长度检查（统一接口） ===
        thought_warnings = self._check_thought_length(sample)
        warnings.extend(thought_warnings)
        
        # === 7. API 调用检查（统一接口） ===
        tool_names = [t.name for t in sample.tools]
        tool_map = {t.name: t for t in sample.tools}
        
        for i, call in enumerate(sample.api_calls):
            call_errors, call_warnings = self._check_api_call(call, i, tool_names, tool_map)
            errors.extend(call_errors)
            warnings.extend(call_warnings)
        
        # === 8. Finish 函数语义检查（统一接口） ===
        finish_errors = self._check_finish_function(sample)
        errors.extend(finish_errors)
        
        return errors, warnings
    
    def _check_tool_definition(self, tool: ToolDefinition, idx: int) -> Tuple[List[str], List[str]]:
        """检查工具定义（统一接口）"""
        errors = []
        warnings = []
        
        if not tool.name:
            errors.append(f"Tool {idx}: missing 'name'")
        
        if not tool.description:
            errors.append(f"Tool {idx} ({tool.name}): missing 'description'")
        
        if tool.parameters is None:
            errors.append(f"Tool {idx} ({tool.name}): missing 'parameters'")
        else:
            for j, param in enumerate(tool.parameters):
                if not param.name:
                    errors.append(f"Tool {idx} ({tool.name}) Parameter {j}: missing 'name'")
                if not param.type:
                    errors.append(f"Tool {idx} ({tool.name}) Parameter {j} ({param.name}): missing 'type'")
        
                # required 和 optional 互补性检查
                # 同为 True 或同为 False 都是错误
                if param.required == param.optional:
                    if param.required and param.optional:
                        errors.append(
                            f"Tool {idx} ({tool.name}) param '{param.name}': "
                            f"INVALID STATE - both required and optional are True"
                        )
                    else:
                        errors.append(
                            f"Tool {idx} ({tool.name}) param '{param.name}': "
                            f"INVALID STATE - both required and optional are False"
                        )
        
        # Finish 函数额外检查
        if tool.name == 'Finish':
            required_params = [p.name for p in (tool.parameters or []) if p.required]
            if 'return_type' not in required_params:
                errors.append(f"Tool {idx} (Finish): 'return_type' should be in required parameters")
        
        return errors, warnings
    
    def _check_action_format(self, call: APICall, idx: int) -> List[str]:
        """
        检查 Assistant Action 格式（从 metadata 读取原始文本）
        """
        errors = []
        
        # 从 metadata 获取原始 assistant 文本
        metadata = call.metadata or {}
        raw_text = metadata.get('raw_assistant_text', '')
        action_input_parsed = metadata.get('action_input_parsed', True)
        
        if not raw_text:
            return errors
        
        # 检查是否有 Thought
        if 'Thought:' not in raw_text and 'thought:' not in raw_text.lower():
            errors.append(f"API Call {idx} ({call.name}): missing 'Thought'")
        
        # 检查是否有 Action
        if 'Action:' not in raw_text and 'action:' not in raw_text.lower():
            errors.append(f"API Call {idx} ({call.name}): missing 'Action'")
        
        # 检查 Action Input JSON 格式
        if not action_input_parsed:
            is_truncated, truncation_reason = self._is_response_truncated(raw_text)
            api_label = f"Finish" if call.name == 'Finish' else f"API '{call.name}'"
            if is_truncated:
                errors.append(f"[TRUNCATED] API Call {idx} {api_label}: Action Input truncated ({truncation_reason})")
            else:
                errors.append(f"API Call {idx} {api_label}: Action Input JSON format error")
        
        return errors
    
    def _is_response_truncated(self, text: str) -> Tuple[bool, str]:
        """检测响应是否被截断"""
        truncation_markers = [
            ('...', 'ends with ellipsis'),
            ('…', 'ends with ellipsis'),
            ('...}', 'truncated JSON'),
        ]
        
        text_stripped = text.strip()
        
        for marker, reason in truncation_markers:
            if text_stripped.endswith(marker):
                return True, reason
        
        # 检查括号不匹配
        open_braces = text.count('{')
        close_braces = text.count('}')
        if open_braces > close_braces:
            return True, f"unclosed braces ({open_braces} open, {close_braces} close)"
        
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        if open_brackets > close_brackets:
            return True, f"unclosed brackets ({open_brackets} open, {close_brackets} close)"
        
        # 检查未闭合的字符串
        quote_count = text.count('"')
        if quote_count % 2 == 1:
            return True, "unclosed string (odd number of quotes)"
        
        return False, ""
    
    def _check_api_call(self, call: APICall, idx: int, 
                        tool_names: List[str], 
                        tool_map: Dict[str, ToolDefinition]) -> Tuple[List[str], List[str]]:
        """检查 API 调用"""
        errors = []
        warnings = []
        
        if not call.name:
            errors.append(f"API Call {idx}: missing 'name'")
            return errors, warnings
        
        if call.arguments is None:
            errors.append(f"API Call {idx} ({call.name}): missing 'arguments'")
        elif not isinstance(call.arguments, dict):
            errors.append(f"API Call {idx} ({call.name}): 'arguments' must be a dict")
        
        # API 必须在工具列表中（Finish 除外）
        if call.name != 'Finish' and call.name not in tool_names:
            errors.append(f"API Call {idx}: API '{call.name}' not in available tools")
        
        # 必填参数检查（Finish 单独处理）
        if call.name != 'Finish' and call.name in tool_map:
            tool = tool_map[call.name]
            provided_args = set(call.arguments.keys()) if call.arguments else set()
            
            for param in tool.parameters:
                if not param.optional and param.name not in provided_args:
                    if param.default is None:
                        errors.append(
                            f"API Call {idx} ({call.name}): missing required parameter '{param.name}'"
                        )
        
        return errors, warnings
    
    def _check_thought_length(self, sample: APIAgentSample) -> List[str]:
        """检查 Thought 长度"""
        warnings = []
        
        for i, call in enumerate(sample.api_calls):
            thought = call.metadata.get('thought', '') if call.metadata else ''
            if thought:
                sentence_count = self._count_sentences(thought)
                if sentence_count > self.MAX_THOUGHT_SENTENCES:
                    warnings.append(
                        f"API Call {i} ({call.name}): Thought exceeds {self.MAX_THOUGHT_SENTENCES} sentences "
                        f"(has {sentence_count})"
                    )
        
        return warnings
    
    def _check_finish_function(self, sample: APIAgentSample) -> List[str]:
        """检查最后一个 Finish 函数的语义正确性"""
        errors = []
        
        # 找最后一个 Finish
        last_finish = None
        last_finish_idx = -1
        for i, call in enumerate(sample.api_calls):
            if call.name == 'Finish':
                last_finish = call
                last_finish_idx = i
        
        if last_finish is None:
            return errors
        
        args = last_finish.arguments or {}
        
        return_type = args.get('return_type')
        if not return_type:
            errors.append(f"Last Finish (Call {last_finish_idx}): missing 'return_type'")
        elif return_type not in ['give_answer', 'give_up_and_restart']:
            errors.append(
                f"Last Finish (Call {last_finish_idx}): invalid return_type='{return_type}' "
                f"(must be 'give_answer' or 'give_up_and_restart')"
            )
        elif return_type == 'give_answer':
            if 'final_answer' not in args:
                errors.append(
                    f"Last Finish (Call {last_finish_idx}): give_answer missing 'final_answer'"
                )
        
        return errors
    
    def _count_sentences(self, text: str) -> int:
        """
        统计句子数量
        核心原则：只统计完整的自然语言推理句子，数据展示列表不算多个句子。
        """
        if not text or not text.strip():
            return 0
        
        # Step 1: 保护代码块
        code_block_pattern = r'```[\s\S]*?```'
        code_blocks = re.findall(code_block_pattern, text)
        for i, block in enumerate(code_blocks):
            text = text.replace(block, f'<CODE{i}>')
        
        # Step 2: 保护连续的编号列表（短列表项）
        numbered_list_pattern = r'(?:(?:^|\n)\s*\d+\.\s+[^\n]+)+'
        numbered_lists = re.findall(numbered_list_pattern, text)
        for i, lst in enumerate(numbered_lists):
            items = re.findall(r'\d+\.\s+([^\n]+)', lst)
            if items:
                avg_len = sum(len(item) for item in items) / len(items)
                if avg_len < 100:
                    text = text.replace(lst, f'<NUMLIST{i}>')
        
        # Step 3: 保护连续的符号列表
        bullet_list_pattern = r'(?:(?:^|\n)\s*[-*]\s+[^\n]+)+'
        bullet_lists = re.findall(bullet_list_pattern, text)
        for i, lst in enumerate(bullet_lists):
            items = re.findall(r'[-*]\s+([^\n]+)', lst)
            if items:
                avg_len = sum(len(item) for item in items) / len(items)
                if avg_len < 100:
                    text = text.replace(lst, f'<BULLETLIST{i}>')
        
        # Step 4: 保护 URL
        url_pattern = r'https?://[^\s\)\]>]+'
        urls = re.findall(url_pattern, text)
        for i, url in enumerate(urls):
            text = text.replace(url, f'<URL{i}>')
        
        # Step 5: 保护域名
        domain_pattern = r'\b(?:www\.)?[a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z0-9][-a-zA-Z0-9]*)*\.(?:com|org|net|edu|gov|io|co)\b'
        domains = re.findall(domain_pattern, text)
        for i, domain in enumerate(domains):
            text = text.replace(domain, f'<DOMAIN{i}>')
        
        # Step 6: 保护 JSON 块
        json_pattern = r'\{[^}]*\}'
        json_blocks = re.findall(json_pattern, text)
        for i, block in enumerate(json_blocks):
            text = text.replace(block, f'<JSON{i}>')
        
        # Step 7: 处理常见缩写
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|etc|vs|e\.g|i\.e)\.\s', r'\1<ABBR> ', text, flags=re.IGNORECASE)
        
        # Step 8: 处理数字中的点
        while re.search(r'(\d+)\.(\d+)', text):
            text = re.sub(r'(\d+)\.(\d+)', r'\1<DOT>\2', text)
        
        # Step 9: 处理编号
        text = re.sub(r'(\d+)\.\s', r'\1<NUM> ', text)
        
        # Step 10: 处理单字母缩写
        text = re.sub(r'\b([A-Z])\.\s', r'\1<LETTER> ', text)
        
        # Step 11: 处理省略号
        text = re.sub(r'\.\.\.+', '.', text)
        
        # Step 12: 分割句子
        sentences = re.split(r'[.!?]+', text)
        
        # Step 13: 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        
        return len(sentences)


