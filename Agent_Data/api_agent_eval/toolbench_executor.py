#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToolBench 数据集执行器

包含 ToolBench 相关的所有检查逻辑：

1. ToolBenchFormatChecker - 格式检查
   - 必需字段检查（id, conversations）
   - id 格式检查（Step N:）
   - conversations 结构检查（system, user, assistant）
   - API 定义格式检查
   - Assistant Action 格式检查（Thought, Action, Action Input）
   - Thought 长度检查（最多5句话）
   - Action Input JSON 格式检查
   - Finish 函数语义检查

2. ToolBenchExecutabilityChecker - 静态可执行性检查
   - System Constraint Check - API 是否在样本定义的工具列表中
   - Toolenv Definition Check - API 是否在全局工具库中存在
   - Required Parameter Check - 必需参数是否完整
   - Train Derivability - final_answer 是否能从 API 响应推导
   - Query-Answer Relevance - final_answer 是否回答了用户查询

3. ToolBenchDynamicChecker - 动态可执行性检查（实际调用 RapidAPI）
   - API 真实存在性 - API 是否真实存在且可访问
   - 参数格式正确性 - 参数格式是否正确
   - 响应有效性 - 是否能获得有效响应
   - 答案可推导性 - final_answer 是否能从真实 API 返回推导
"""
import os
import re
import json
import glob
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

from data_types import APIAgentSample, ToolDefinition, APICall, Parameter
from api_executor import (
    FormatChecker, ExecutabilityChecker, DynamicChecker,
    register_format_checker, register_executability_checker, register_dynamic_checker
)


# =============================================================================
# LLM 配置（用于 LLM Judge）
# =============================================================================

LLM_API_KEY = 'sk-o0QqcwC8XNHU6gGT7CYdMSQGJQQMtjKJSqw6K9G21IaoOElt'
LLM_BASE_URL = 'http://35.220.164.252:3888/v1/'
LLM_MODEL = 'gpt-4.1'


# =============================================================================
# LLM Judge Prompt 模板
# =============================================================================

DERIVABILITY_PROMPT = """你是一个数据质量评估专家。请判断以下 Agent 最终答案是否可以从 API 响应中推导出来。

【API 响应】
{api_responses}

【最终答案】
{final_answer}

请判断：最终答案中的关键信息是否能在 API 响应中找到支持？是否存在"编造"或"幻觉"的内容？

判断标准：
- 只要答案如实反映了 API 响应的内容，就应该判定为 derivable=true
- 如果 API 返回空结果，答案如实说明"没有数据"也是正确的
- 只有当答案包含 API 响应中完全不存在的信息时，才判定为 derivable=false

请以 JSON 格式输出：
```json
{{
    "derivable": true/false,
    "reason": "简要说明判断理由"
}}
```

只输出 JSON，不要其他内容。"""


RELEVANCE_PROMPT = """你是一个数据质量评估专家。请判断以下 Agent 最终答案是否正确回答了用户的查询。

【用户查询】
{query}

【最终答案】
{final_answer}

请判断：最终答案是否回应了用户的问题？

判断标准：
- 只要答案针对用户的问题给出了相关回应，就应该判定为 relevant=true
- 如果查询的数据不存在，答案如实说明"没有数据"也算正确回答
- 只有当答案完全偏题、答非所问时，才判定为 relevant=false

请以 JSON 格式输出：
```json
{{
    "relevant": true/false,
    "reason": "简要说明判断理由"
}}
```

只输出 JSON，不要其他内容。"""


# =============================================================================
# ToolBench 格式检查器
# =============================================================================

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


# =============================================================================
# ToolBench 可执行性检查器
# =============================================================================

class ToolBenchExecutabilityChecker(ExecutabilityChecker):
    """
    ToolBench 数据集可执行性检查器
    
    复用自 evaluate_toolbench_basic.py.evaluate_executability
    
    检查项：
    1. System Constraint Check（系统约束检查）
       - API 调用是否在样本定义的工具列表中
       - Finish 函数跳过此检查
       
    2. Toolenv Definition Check（工具库定义检查）[可选]
       - API 是否在 toolenv 全局工具库中存在
       - 需要提供 toolenv_path
       
    3. Required Parameter Check（必需参数检查）
       - API 调用时是否提供了所有必需参数
       
    4. Train Derivability（训练数据可推导性）
       - final_answer 是否能从 API 响应中推导
       
    5. Query-Answer Relevance（查询-回答相关性）
       - final_answer 是否回答了用户的 query
    """
    
    def __init__(self, 
                 toolenv_path: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        初始化 ToolBench 可执行性检查器
        
        Args:
            toolenv_path: toolenv 工具库路径，用于全局 API 存在性检查
                         如果为 None，则跳过 toolenv 检查
            cache_dir: API 映射缓存目录，默认为当前目录下的 cache/
        """
        self.toolenv_path = toolenv_path
        # 默认使用当前目录下的 cache 目录
        if cache_dir is None:
            import os
            self.cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        else:
            self.cache_dir = cache_dir
        
        # API 映射表：{标准化API名: {url, method, host, required_parameters, ...}}
        self.api_mapping: Dict[str, Dict[str, Any]] = {}
        self._mapping_loaded = False
    
    def _build_api_mapping(self, force_rebuild: bool = False) -> None:
        """
        从 toolenv 目录构建 API 映射表（带缓存）
        
        复用自 evaluate_toolbench_basic.py._build_api_mapping
        
        映射格式：
        {
            "{api_name}_for_{tool_name_normalized}": {
                "url": "https://...",
                "method": "GET/POST",
                "host": "xxx.p.rapidapi.com",
                "required_parameters": [...],
                "optional_parameters": [...],
                "tool_name": "原始工具名",
                "api_name": "原始API名"
            }
        }
        """
        if self._mapping_loaded and not force_rebuild:
            return
        
        if not self.toolenv_path:
            return
        
        # 缓存路径
        cache_path = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.cache_dir, 'api_mapping_cache.json')
        
        # 尝试从缓存加载
        if cache_path and not force_rebuild and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.api_mapping = json.load(f)
                self._mapping_loaded = True
                return
            except Exception:
                pass
        
        if not os.path.exists(self.toolenv_path):
            return
        
        # 遍历所有类别目录下的 JSON 文件
        json_files = glob.glob(os.path.join(self.toolenv_path, '**', '*.json'), recursive=True)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    tool_data = json.load(f)
                
                tool_name = tool_data.get('tool_name', '')
                host = tool_data.get('host', '')
                
                # 将工具名标准化为下划线格式
                tool_name_normalized = re.sub(r'[^a-zA-Z0-9]+', '_', tool_name).lower().strip('_')
                
                for api in tool_data.get('api_list', []):
                    api_name = api.get('name', '')
                    api_name_normalized = re.sub(r'[^a-zA-Z0-9]+', '_', api_name).lower().strip('_')
                    
                    # 构建完整的 API 键
                    full_api_key = f"{api_name_normalized}_for_{tool_name_normalized}"
                    
                    self.api_mapping[full_api_key] = {
                        'url': api.get('url', ''),
                        'method': api.get('method', 'GET'),
                        'host': host,
                        'required_parameters': api.get('required_parameters', []),
                        'optional_parameters': api.get('optional_parameters', []),
                        'tool_name': tool_name,
                        'api_name': api_name,
                        'code': api.get('code', ''),
                    }
            except Exception:
                pass
        
        self._mapping_loaded = True
        
        # 保存到缓存
        if cache_path:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.api_mapping, f, ensure_ascii=False)
            except Exception:
                pass
    
    def check(self, sample: APIAgentSample) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """检查 ToolBench 样本的可执行性"""
        errors = []
        warnings = []
        stats = {
            'api_calls_checked': 0,
            'toolenv_check_enabled': bool(self.toolenv_path),
            'train_derivability': None,
            'query_relevance': None,
        }
        
        # 构建工具映射（如果提供了 toolenv_path）
        if self.toolenv_path:
            self._build_api_mapping()
        
        # 1. 构建样本内的工具名映射
        tool_names = [t.name for t in sample.tools]
        tool_map = {t.name: t for t in sample.tools}
        
        # 2. 检查每个 API 调用
        for i, call in enumerate(sample.api_calls):
            if not call.name:
                continue
            
            stats['api_calls_checked'] += 1
            
            # Finish 函数跳过
            if call.name == 'Finish':
                continue
            
            # 2.1 System Constraint Check - API 是否在样本工具列表中
            if call.name not in tool_names:
                errors.append(f"API Call {i}: API '{call.name}' not defined in sample tools")
                continue
            
            # 2.2 Toolenv Definition Check - API 是否在全局工具库中
            if self.toolenv_path and self.api_mapping:
                api_key = call.name.lower()
                if api_key not in self.api_mapping:
                    errors.append(f"API Call {i}: API '{call.name}' not found in toolenv mapping")
            
            # 2.3 Required Parameter Check (Sample Tool) - 基于样本内工具定义的必需参数检查
            if call.name in tool_map:
                tool = tool_map[call.name]
                provided_args = set(call.arguments.keys()) if call.arguments else set()
                
                for param in (tool.parameters or []):
                    if not param.optional and param.name not in provided_args:
                        if param.default is None:
                            errors.append(
                                f"API Call {i} ({call.name}): missing required parameter '{param.name}' (sample tool)"
                            )
            
            # 2.4 Required Parameter Check (Toolenv) - 基于全局工具库的必需参数检查
            if self.toolenv_path and self.api_mapping:
                api_key = call.name.lower()
                if api_key in self.api_mapping:
                    api_info = self.api_mapping[api_key]
                    provided_args = set(call.arguments.keys()) if call.arguments else set()
                    
                    # 检查 toolenv 中定义的 required_parameters
                    for req_param in api_info.get('required_parameters', []):
                        param_name = req_param.get('name', '') if isinstance(req_param, dict) else str(req_param)
                        # 获取 default 值（toolenv 格式：{"name": "...", "default": "..."}）
                        param_default = req_param.get('default', '') if isinstance(req_param, dict) else ''
                        
                        # 只有当：1) 参数未提供 且 2) 没有 default 值时，才报错
                        if param_name and param_name not in provided_args:
                            if not param_default:  # default 为空字符串或 None 时才报错
                                errors.append(
                                    f"API Call {i} ({call.name}): missing required parameter '{param_name}' (toolenv)"
                                )
        
        # 3. Train Derivability - 检查 final_answer 是否能从 API 响应推导（LLM Judge）
        if sample.final_answer:
            # 收集所有 API 响应
            all_responses = []
            for call in sample.api_calls:
                if call.name != 'Finish' and call.response:
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
        
        # 4. Query-Answer Relevance - 检查 final_answer 是否回答了 query（LLM Judge）
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
    
    def _check_keyword_derivability(self, source_text: str, target_text: str) -> Tuple[bool, str]:
        """
        关键词匹配检查答案可推导性
        
        复用自 evaluate_toolbench_basic.py._check_keyword_derivability
        
        从 source_text 中提取关键信息（数字、专有名词、关键词），
        检查这些信息是否出现在 target_text 中。
        
        Args:
            source_text: 源文本（如 final_answer 或 query）
            target_text: 目标文本（如 API 响应或 final_answer）
        
        Returns:
            (derivable: bool, info: str)
        """
        # 1. 提取数字（包括小数、百分比、金额等）
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|°[CF]?)?\b', source_text)
        
        # 2. 提取可能的专有名词（连续大写字母开头的词）
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', source_text)
        
        # 3. 提取引号中的内容
        quoted = re.findall(r'["\']([^"\']+)["\']', source_text)
        
        # 4. 提取特殊格式（日期等）
        dates = re.findall(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b', source_text)
        
        # 合并所有关键信息
        all_keywords = []
        all_keywords.extend([(n, 'number') for n in numbers if len(n) >= 2])
        all_keywords.extend([(n, 'proper_noun') for n in proper_nouns if len(n) >= 3])
        all_keywords.extend([(q, 'quoted') for q in quoted if len(q) >= 3])
        all_keywords.extend([(d, 'date') for d in dates])
        
        if not all_keywords:
            return True, "No extractable keywords (assumed derivable)"
        
        # 检查匹配
        matched = []
        not_matched = []
        
        target_text_lower = target_text.lower()
        
        for keyword, kw_type in all_keywords:
            if keyword.lower() in target_text_lower:
                matched.append((keyword, kw_type))
            else:
                not_matched.append((keyword, kw_type))
        
        # 计算匹配率
        match_rate = len(matched) / len(all_keywords) if all_keywords else 0
        
        # 判断是否可推导：至少 50% 的关键信息能匹配到
        derivable = match_rate >= 0.5
        
        info = f"Matched {len(matched)}/{len(all_keywords)} ({match_rate*100:.0f}%)"
        if matched:
            info += f" | Found: {[k for k, _ in matched[:5]]}"
        if not_matched:
            info += f" | Missing: {[k for k, _ in not_matched[:5]]}"
        
        return derivable, info
    
    def _check_derivability_llm(
        self, 
        final_answer: str, 
        api_responses: str,
        max_retries: int = 3
    ) -> Tuple[bool, str]:
        """
        使用 LLM 作为 Judge 检查答案可推导性
        
        Args:
            final_answer: 最终答案
            api_responses: API 响应文本
            max_retries: 最大重试次数
        
        Returns:
            (derivable: bool, reason: str)
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
                
                # 提取 JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                derivable = result.get('derivable', False)
                reason = result.get('reason', '')
                
                return derivable, reason
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return False, "LLM call failed"
    
    def _check_relevance_llm(
        self, 
        query: str, 
        final_answer: str,
        max_retries: int = 3
    ) -> Tuple[bool, str]:
        """
        使用 LLM 作为 Judge 检查查询-答案相关性
        
        Args:
            query: 用户查询
            final_answer: 最终答案
            max_retries: 最大重试次数
        
        Returns:
            (relevant: bool, reason: str)
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
                
                # 提取 JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                relevant = result.get('relevant', False)
                reason = result.get('reason', '')
                
                return relevant, reason
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return False, "LLM call failed"


# =============================================================================
# ToolBench 动态可执行性检查器
# =============================================================================

class ToolBenchDynamicChecker(DynamicChecker):
    """
    ToolBench 数据集动态可执行性检查器
    
    通过实际调用 RapidAPI 来验证：
    1. API 是否真实存在且可访问
    2. 参数格式是否正确
    3. 是否能获得有效响应
    4. final_answer 是否能从真实 API 返回推导出来
    
    Pass Rate 计算规则（符合 ToolBench 论文定义）：
    - 所有 API 调用都成功返回（非错误响应）
    - 最终 Finish 的 return_type = "give_answer"
    
    注意：此检查器需要 RapidAPI Key 和网络访问权限。
    
    使用方式:
        checker = ToolBenchDynamicChecker(
            rapidapi_key='your_key',
            toolenv_path='/path/to/toolenv/tools',
            cache_dir='/path/to/cache'
        )
        
        results = checker.check_sample(sample)
        # 或批量检查
        results = checker.check_batch(samples, sample_size=100)
    """
    
    def __init__(self,
                 rapidapi_key: Optional[str] = None,
                 toolenv_path: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 timeout: int = 180):  # 默认 3 分钟超时
        """
        初始化动态检查器
        
        Args:
            rapidapi_key: RapidAPI Key，如果为 None 则尝试从环境变量读取
            toolenv_path: toolenv 工具库路径
            cache_dir: API 映射缓存目录
            timeout: 单个 API 调用的超时时间（秒），默认 180 秒（3 分钟）
        """
        # 获取 API Key
        self.rapidapi_key = rapidapi_key or os.environ.get('RAPIDAPI_KEY')
        
        self.toolenv_path = toolenv_path
        self.cache_dir = cache_dir
        self.timeout = timeout
        
        # API 映射表
        self.api_mapping: Dict[str, Dict[str, Any]] = {}
        self._mapping_loaded = False
    
    def _build_api_mapping(self, force_rebuild: bool = False) -> None:
        """
        从 toolenv 目录构建 API 映射表（带缓存）
        """
        if self._mapping_loaded and not force_rebuild:
            return
        
        if not self.toolenv_path:
            return
        
        # 缓存路径
        cache_path = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.cache_dir, 'api_mapping_cache.json')
        
        # 尝试从缓存加载
        if cache_path and not force_rebuild and os.path.exists(cache_path):
            try:
                print(f"从缓存加载 API 映射: {cache_path}", flush=True)
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.api_mapping = json.load(f)
                print(f"已加载 {len(self.api_mapping):,} 个 API 定义", flush=True)
                self._mapping_loaded = True
                return
            except Exception as e:
                print(f"缓存加载失败: {e}，重新构建...", flush=True)
        
        if not os.path.exists(self.toolenv_path):
            print(f"toolenv 路径不存在: {self.toolenv_path}", flush=True)
            return
        
        # 遍历所有 JSON 文件
        print(f"从 toolenv 构建 API 映射: {self.toolenv_path}", flush=True)
        json_files = glob.glob(os.path.join(self.toolenv_path, '**', '*.json'), recursive=True)
        print(f"找到 {len(json_files):,} 个 JSON 文件", flush=True)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    tool_data = json.load(f)
                
                tool_name = tool_data.get('tool_name', '')
                host = tool_data.get('host', '')
                
                tool_name_normalized = re.sub(r'[^a-zA-Z0-9]+', '_', tool_name).lower().strip('_')
                
                for api in tool_data.get('api_list', []):
                    api_name = api.get('name', '')
                    api_name_normalized = re.sub(r'[^a-zA-Z0-9]+', '_', api_name).lower().strip('_')
                    
                    full_api_key = f"{api_name_normalized}_for_{tool_name_normalized}"
                    
                    self.api_mapping[full_api_key] = {
                        'url': api.get('url', ''),
                        'method': api.get('method', 'GET'),
                        'host': host,
                        'required_parameters': api.get('required_parameters', []),
                        'optional_parameters': api.get('optional_parameters', []),
                        'tool_name': tool_name,
                        'api_name': api_name,
                    }
            except Exception:
                pass
        
        self._mapping_loaded = True
        print(f"已构建 {len(self.api_mapping):,} 个 API 定义", flush=True)
        
        # 保存到缓存
        if cache_path:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.api_mapping, f, ensure_ascii=False)
                print(f"已保存缓存到: {cache_path}", flush=True)
            except Exception as e:
                print(f"保存缓存失败: {e}", flush=True)
    
    def _call_rapidapi(self, api_info: Dict, params: Dict) -> Tuple[bool, Optional[str], Any]:
        """
        调用 RapidAPI
        
        Args:
            api_info: API 定义信息
            params: 调用参数
        
        Returns:
            (success, error_message, response_data)
        """
        import requests
        
        if not self.rapidapi_key:
            return False, "No RapidAPI key configured", None
        
        url = api_info.get('url', '')
        method = api_info.get('method', 'GET').upper()
        host = api_info.get('host', '')
        
        if not url or not host:
            return False, "Missing URL or host in API definition", None
        
        headers = {
            "X-RapidAPI-Key": self.rapidapi_key,
            "X-RapidAPI-Host": host
        }
        
        # 代理配置
        http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
        https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
        proxies = None
        if http_proxy or https_proxy:
            proxies = {
                'http': http_proxy,
                'https': https_proxy
            }
        
        try:
            # 处理 URL 中的路径参数
            for param_name, param_value in params.items():
                url = url.replace(f'{{{param_name}}}', str(param_value))
            
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, 
                                       proxies=proxies, timeout=self.timeout)
            elif method == 'POST':
                headers['Content-Type'] = 'application/json'
                response = requests.post(url, headers=headers, json=params,
                                        proxies=proxies, timeout=self.timeout)
            else:
                return False, f"Unsupported HTTP method: {method}", None
            
            # 检查响应状态
            if response.status_code == 200:
                try:
                    return True, None, response.json()
                except:
                    return True, None, response.text
            elif response.status_code == 401:
                return False, "Unauthorized (invalid API key)", None
            elif response.status_code == 403:
                return False, "Forbidden (API access denied)", None
            elif response.status_code == 404:
                return False, "API endpoint not found", None
            elif response.status_code == 429:
                return False, "Rate limit exceeded", None
            elif response.status_code >= 500:
                return False, f"Server error ({response.status_code})", None
            else:
                return False, f"HTTP {response.status_code}: {response.text[:200]}", None
                
        except Exception as e:
            if 'Timeout' in str(type(e).__name__):
                return False, "Request timeout", None
            elif 'ConnectionError' in str(type(e).__name__):
                return False, "Connection error", None
            else:
                return False, f"Request error: {str(e)}", None
    
    def _check_derivability_llm(
        self, 
        final_answer: str, 
        api_responses: str,
        max_retries: int = 3
    ) -> Tuple[bool, str]:
        """
        使用 LLM 作为 Judge 检查答案可推导性
        
        Args:
            final_answer: 最终答案
            api_responses: API 响应文本
            max_retries: 最大重试次数
        
        Returns:
            (derivable: bool, reason: str)
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
                
                # 提取 JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                derivable = result.get('derivable', False)
                reason = result.get('reason', '')
                
                return derivable, reason
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return False, "LLM call failed"
    
    def check_sample(self, sample: APIAgentSample) -> Dict[str, Any]:
        """
        检查单个样本的动态可执行性
        
        Args:
            sample: API Agent 样本
            
        Returns:
            检查结果字典
        """
        if not self.rapidapi_key:
            return {
                'error': 'No RapidAPI key configured',
                'passed': False,
            }
        
        # 构建 API 映射
        if self.toolenv_path:
            self._build_api_mapping()
        
        result = {
            'sample_id': sample.sample_id,
            'passed': True,
            'api_calls': [],
            'finish_type': None,
            'final_answer': sample.final_answer,
            'derivability': None,
            'errors': [],
        }
        
        all_api_responses = []
        
        for i, call in enumerate(sample.api_calls):
            if not call.name:
                continue
            
            # 处理 Finish
            if call.name == 'Finish':
                args = call.arguments or {}
                return_type = args.get('return_type', '')
                
                if return_type == 'give_answer':
                    result['finish_type'] = 'give_answer'
                elif return_type == 'give_up_and_restart':
                    result['finish_type'] = 'give_up_and_restart'
                    result['passed'] = False
                    result['errors'].append("Finish: give_up_and_restart")
                else:
                    result['finish_type'] = 'invalid'
                continue
            
            # 查找 API 定义
            api_key = call.name.lower()
            api_info = self.api_mapping.get(api_key)
            
            if not api_info:
                # 模糊匹配
                for key in self.api_mapping:
                    if key.startswith(api_key) or api_key.startswith(key):
                        api_info = self.api_mapping[key]
                        break
            
            call_result = {
                'api_name': call.name,
                'arguments': call.arguments,
                'success': False,
                'error': None,
                'response': None,
            }
            
            if not api_info:
                call_result['error'] = 'API not found in toolenv mapping'
                result['passed'] = False
                result['errors'].append(f"API '{call.name}' not found in toolenv")
            else:
                # 调用 API
                success, error, response = self._call_rapidapi(
                    api_info,
                    call.arguments if isinstance(call.arguments, dict) else {}
                )
                
                call_result['success'] = success
                call_result['error'] = error
                call_result['response'] = response
                
                if success:
                    response_str = json.dumps(response, ensure_ascii=False) if isinstance(response, (dict, list)) else str(response)
                    all_api_responses.append({
                        'api_name': call.name,
                        'response': response_str  # 不截断，保留完整响应
                    })
                else:
                    result['passed'] = False
                    result['errors'].append(f"API '{call.name}': {error}")
            
            result['api_calls'].append(call_result)
        
        # 检查 finish_type
        if result['finish_type'] != 'give_answer':
            result['passed'] = False
        
        # 检查答案可推导性（LLM Judge）
        if result['passed'] and sample.final_answer and all_api_responses:
            all_responses_text = "\n".join([
                f"[{r['api_name']}]: {r['response']}" for r in all_api_responses
            ])
            
            derivable, reason = self._check_derivability_llm(
                sample.final_answer, all_responses_text
            )
            
            result['derivability'] = {
                'derivable': derivable,
                'reason': reason,
            }
        
        return result
    
    def check_batch(self, samples: List[APIAgentSample], 
                    sample_size: int = 0,
                    show_progress: bool = True) -> Dict[str, Any]:
        """
        批量检查样本的动态可执行性
        
        Args:
            samples: API Agent 样本列表
            sample_size: 抽样数量，0 表示全部
            show_progress: 是否显示进度
            
        Returns:
            批量检查结果
        """
        if not self.rapidapi_key:
            return {
                'error': 'No RapidAPI key configured',
                'total_samples': 0,
                'passed_samples': 0,
                'pass_rate': 0.0,
            }
        
        # 构建 API 映射
        if self.toolenv_path:
            self._build_api_mapping()
        
        # 选择样本
        if sample_size > 0:
            test_samples = samples[:min(sample_size, len(samples))]
        else:
            test_samples = samples
        
        results = {
            'total_samples': len(test_samples),
            'passed_samples': 0,
            'failed_samples': 0,
            'api_call_stats': {
                'total': 0,
                'success': 0,
                'failed': 0,
                'not_found': 0,
            },
            'finish_stats': {
                'give_answer': 0,
                'give_up_and_restart': 0,
                'invalid': 0,
            },
            'derivability': {
                'total_checked': 0,
                'derivable': 0,
                'not_derivable': 0,
            },
            'failed_apis': Counter(),
            'success_apis': Counter(),
            'errors': [],
            'sample_results': [],
        }
        
        iterator = test_samples
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(test_samples, desc="Verifying API calls")
            except ImportError:
                pass
        
        for sample in iterator:
            sample_result = self.check_sample(sample)
            results['sample_results'].append(sample_result)
            
            if sample_result.get('passed'):
                results['passed_samples'] += 1
            else:
                results['failed_samples'] += 1
                if sample_result.get('errors'):
                    results['errors'].append({
                        'sample_id': sample_result['sample_id'],
                        'errors': sample_result['errors']
                    })
            
            # 统计 API 调用
            for call_result in sample_result.get('api_calls', []):
                results['api_call_stats']['total'] += 1
                if call_result['success']:
                    results['api_call_stats']['success'] += 1
                    results['success_apis'][call_result['api_name']] += 1
                elif call_result['error'] == 'API not found in toolenv mapping':
                    results['api_call_stats']['not_found'] += 1
                else:
                    results['api_call_stats']['failed'] += 1
                    results['failed_apis'][call_result['api_name']] += 1
            
            # 统计 finish
            finish_type = sample_result.get('finish_type')
            if finish_type in results['finish_stats']:
                results['finish_stats'][finish_type] += 1
            
            # 统计 derivability
            deriv = sample_result.get('derivability')
            if deriv:
                results['derivability']['total_checked'] += 1
                if deriv['derivable']:
                    results['derivability']['derivable'] += 1
                else:
                    results['derivability']['not_derivable'] += 1
        
        # 计算 pass rate
        results['pass_rate'] = (
            results['passed_samples'] / results['total_samples'] * 100
            if results['total_samples'] > 0 else 0.0
        )
        
        # 计算 derivability rate
        if results['derivability']['total_checked'] > 0:
            results['derivability']['rate'] = (
                results['derivability']['derivable'] / 
                results['derivability']['total_checked'] * 100
            )
        else:
            results['derivability']['rate'] = 0.0
        
        return results


# =============================================================================
# 注册到全局注册表
# =============================================================================

register_format_checker('toolbench', ToolBenchFormatChecker)
register_executability_checker('toolbench', ToolBenchExecutabilityChecker)
register_dynamic_checker('toolbench', ToolBenchDynamicChecker)

