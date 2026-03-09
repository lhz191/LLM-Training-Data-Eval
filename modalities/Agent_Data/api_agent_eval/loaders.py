#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Agent Data Loaders

数据加载器，将不同格式的数据集转换为统一的 APIAgentSample 格式。

支持：
- ToolBench: 多轮对话格式，有真实 API 响应
- xLAM-60k: 单轮调用格式，只有工具定义
"""

import json
import re
import ast
from typing import List, Dict, Any, Optional, Tuple, Iterator
from tqdm import tqdm

from data_types import (
    Parameter,
    ToolDefinition,
    APICall,
    APIAgentSample,
)


# =============================================================================
# Base Loader
# =============================================================================

class BaseLoader:
    """数据集加载器基类"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load(self) -> List[APIAgentSample]:
        """加载数据集，返回 APIAgentSample 列表"""
        return list(self.iterate())
    
    def iterate(self) -> Iterator[APIAgentSample]:
        """迭代返回 APIAgentSample，子类需实现"""
        raise NotImplementedError


# =============================================================================
# ToolBench Loader
# =============================================================================

class ToolBenchLoader(BaseLoader):
    """
    ToolBench 数据集加载器
    
    将 ToolBench 的多轮对话格式转换为统一的 APIAgentSample。
    
    ToolBench 数据格式：
    {
        "id": "Step N: 用户指令内容",
        "conversations": [
            {"from": "system", "value": "系统提示 + API定义"},
            {"from": "user", "value": "用户指令"},
            {"from": "assistant", "value": "Thought: ... Action: ... Action Input: ..."},
            {"from": "function", "value": "{'error': '', 'response': ...}"},
            ...
        ]
    }
    """
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data: List[Dict] = []
    
    # =========================================================================
    # ToolBench 解析辅助方法（静态方法）
    # =========================================================================
    
    @staticmethod
    def _extract_balanced_braces(text: str) -> str:
        """提取平衡的大括号内容，处理嵌套和字符串内的大括号"""
        if not text or text[0] != '{':
            return ""
        
        depth = 0
        in_string = False
        escape = False
        
        for i, char in enumerate(text):
            if escape:
                escape = False
                continue
            
            if char == '\\':
                escape = True
                continue
            
            if char == '"' and not escape:
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[:i+1]
        
        return text
    
    @staticmethod
    def _fix_json_newlines(json_str: str) -> str:
        """处理嵌套 JSON 解析时产生的换行符问题"""
        result = []
        in_string = False
        i = 0
        
        while i < len(json_str):
            char = json_str[i]
            
            if char == '\\' and i + 1 < len(json_str):
                result.append(char)
                result.append(json_str[i + 1])
                i += 2
                continue
            
            if char == '"':
                in_string = not in_string
                result.append(char)
                i += 1
                continue
            
            if in_string and char == '\n':
                result.append('\\n')
            elif in_string and char == '\r':
                result.append('\\r')
            elif in_string and char == '\t':
                result.append('\\t')
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)
    
    @staticmethod
    def _is_valid_api_name(name: str) -> bool:
        """判断是否是有效的 API 名"""
        if not name:
            return False
        if re.match(r'^\d+\.?$', name):
            return False
        if name.lower() in ['call', 'use', 'invoke', 'execute', 'run', 'the', 'a', 'an', 'to', 'for', 'with']:
            return False
        if re.match(r'^[\-\*\>\#\.\,\!\?\:\;\(\)\[\]\{\}]+$', name):
            return False
        if len(name) <= 2 and '_' not in name:
            return False
        if not re.search(r'[a-zA-Z]', name):
            return False
        return True
    
    @staticmethod
    def _parse_system_apis(system_text: str) -> List[Dict]:
        """从 system prompt 中解析 API 定义列表"""
        apis = []
        
        marker = "Specifically, you have access to the following APIs:"
        start = system_text.find(marker)
        if start == -1:
            return apis
        
        api_text = system_text[start + len(marker):].strip()
        
        try:
            apis = ast.literal_eval(api_text)
        except:
            try:
                apis = json.loads(api_text)
            except:
                pass
        
        return apis if isinstance(apis, list) else []
    
    @staticmethod
    def _parse_action(assistant_text: str) -> Tuple[Optional[str], Optional[Dict]]:
        """从 assistant 回复中解析 Action 和 Action Input"""
        action_name = None
        action_input = None
        
        action_matches = re.findall(r'Action:\s*(\S+)', assistant_text)
        
        for match in reversed(action_matches):
            candidate = match.strip()
            if ToolBenchLoader._is_valid_api_name(candidate):
                action_name = candidate
                break
        
        if action_name is None and action_matches:
            action_name = action_matches[-1].strip()
        
        input_start = assistant_text.find('Action Input:')
        if input_start != -1:
            brace_start = assistant_text.find('{', input_start)
            if brace_start != -1:
                input_str = ToolBenchLoader._extract_balanced_braces(assistant_text[brace_start:])
                if input_str:
                    input_str_fixed = ToolBenchLoader._fix_json_newlines(input_str)
                    
                    try:
                        action_input_direct = json.loads(input_str)
                        action_input_fixed = json.loads(input_str_fixed)
                        assert action_input_direct == action_input_fixed
                        action_input = action_input_direct
                    except json.JSONDecodeError:
                        try:
                            action_input = json.loads(input_str_fixed)
                        except:
                            try:
                                action_input = ast.literal_eval(input_str)
                            except:
                                pass
        
        return action_name, action_input
    
    @staticmethod
    def _extract_thought(assistant_text: str) -> str:
        """从 assistant 回复中提取 Thought 部分"""
        match = re.search(r'Thought:\s*(.*?)(?:Action:|$)', assistant_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
    
    @staticmethod
    def _extract_step_number(id_str: str) -> Optional[int]:
        """从 id 中提取 Step 数字"""
        if id_str.startswith('Step '):
            match = re.match(r'Step (\d+):', id_str)
            if match:
                return int(match.group(1))
        return None
    
    @staticmethod
    def _extract_query(id_str: str) -> str:
        """从 id 中提取用户指令"""
        if ':' in id_str:
            return ':'.join(id_str.split(':')[1:]).strip()
        return id_str
    
    @staticmethod
    def _parse_function_response(func_value: str) -> Optional[str]:
        """解析 function 响应，直接返回原始字符串"""
        return func_value if func_value else None
    
    # =========================================================================
    # 数据加载和解析方法
    # =========================================================================
    
    def load(self) -> List[Dict]:
        """加载原始 JSON 数据"""
        print(f"📂 Loading ToolBench dataset: {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"✅ Loaded {len(self.data):,} records")
        return self.data
    
    def parse_all(self, show_progress: bool = True) -> List[APIAgentSample]:
        """
        解析所有记录为 APIAgentSample
        
        Args:
            show_progress: 是否显示进度条
        
        Returns:
            APIAgentSample 列表
        """
        if not self.data:
            self.load()
        
        samples = []
        iterator = tqdm(self.data, desc="Parsing ToolBench") if show_progress else self.data
        
        for idx, record in enumerate(iterator):
            sample = self.parse_record(record, idx)
            if sample:
                samples.append(sample)
        
        print(f"✅ Parsed {len(samples):,} samples")
        return samples
    
    def iterate(self, show_progress: bool = True) -> Iterator[APIAgentSample]:
        """
        迭代返回 APIAgentSample（惰性加载，节省内存）
        
        Args:
            show_progress: 是否显示进度条
        
        Yields:
            APIAgentSample
        """
        if not self.data:
            self.load()
        
        iterator = tqdm(self.data, desc="Parsing ToolBench") if show_progress else self.data
        
        for idx, record in enumerate(iterator):
            sample = self.parse_record(record, idx)
            if sample:
                yield sample
    
    def parse_record(self, record: Dict, idx: int = 0) -> Optional[APIAgentSample]:
        """
        解析单条 ToolBench 记录
        
        Args:
            record: 原始记录
            idx: 记录索引
        
        Returns:
            APIAgentSample 或 None（解析失败时）
        """
        try:
            # === 1. 解析 ID 和 Query ===
            id_str = record.get('id', '')
            step_number = self._extract_step_number(id_str)
            query = self._extract_query(id_str)
            
            conversations = record.get('conversations', [])
            if not conversations:
                return None
            
            # === 2. 从 system prompt 解析工具定义 ===
            tools = []
            for conv in conversations:
                if conv.get('from') == 'system':
                    api_dicts = self._parse_system_apis(conv.get('value', ''))
                    tools = [self._dict_to_tool_definition(api) for api in api_dicts]
                    tools = [t for t in tools if t is not None]
                    break
            
            # === 3. 从 user 获取 query（如果 id 中没有） ===
            if not query:
                for conv in conversations:
                    if conv.get('from') == 'user':
                        query = conv.get('value', '').strip()
                        break
            
            # === 4. 解析所有 API 调用 ===
            api_calls = []
            final_answer = None
            
            # 用于 format_check 的原始数据
            raw_assistant_texts = []  # 保存原始 assistant 文本（用于检查 Thought/Action 格式）
            roles_in_conversations = []  # 保存 conversations 中的角色列表
            
            for conv in conversations:
                role = conv.get('from', '')
                roles_in_conversations.append(role)
            
            # 遍历对话，配对 assistant 和 function
            i = 0
            while i < len(conversations):
                conv = conversations[i]
                
                if conv.get('from') == 'assistant':
                    assistant_text = conv.get('value', '')
                    raw_assistant_texts.append(assistant_text)  # 保存原始文本
                    
                    action_name, action_input = self._parse_action(assistant_text)
                    thought = self._extract_thought(assistant_text)
                    
                    if action_name:
                        # 查找下一个 function 响应
                        response = None
                        
                        if i + 1 < len(conversations) and conversations[i + 1].get('from') == 'function':
                            func_value = conversations[i + 1].get('value', '')
                            response = self._parse_function_response(func_value)
                        
                        # 检查是否是 Finish - 也记录为 api_call（用于 format_check）
                        if action_name == 'Finish':
                            if action_input and isinstance(action_input, dict):
                                if action_input.get('return_type') == 'give_answer':
                                    final_answer = action_input.get('final_answer', '')
                            
                            # Finish 也作为 API 调用记录
                            api_call = APICall(
                                name='Finish',
                                arguments=action_input if isinstance(action_input, dict) else {},
                                response=response,
                                metadata={
                                    'thought': thought,
                                    'raw_assistant_text': assistant_text,  # 保存原始文本用于 format_check
                                    'action_input_parsed': action_input is not None,  # 是否成功解析
                                } if thought else {
                                    'raw_assistant_text': assistant_text,
                                    'action_input_parsed': action_input is not None,
                                }
                            )
                            api_calls.append(api_call)
                        else:
                            # 普通 API 调用
                            api_call = APICall(
                                name=action_name,
                                arguments=action_input if isinstance(action_input, dict) else {},
                                response=response,
                                metadata={
                                    'thought': thought,
                                    'raw_assistant_text': assistant_text,
                                    'action_input_parsed': action_input is not None,
                                } if thought else {
                                    'raw_assistant_text': assistant_text,
                                    'action_input_parsed': action_input is not None,
                                }
                            )
                            api_calls.append(api_call)
                
                i += 1
            
            # === 5. 构建 APIAgentSample ===
            sample = APIAgentSample(
                query=query or '',
                tools=tools,
                api_calls=api_calls,
                final_answer=final_answer,
                sample_id=f"toolbench_{idx}",
                source_dataset='toolbench',
                metadata={
                    # 基本信息
                    'step_number': step_number,
                    'original_id': id_str,
                    # 用于 format_check 的原始数据
                    'raw_assistant_texts': raw_assistant_texts,  # 原始 assistant 文本列表
                    'roles_in_conversations': roles_in_conversations,  # 角色列表
                }
            )
            
            return sample
            
        except Exception as e:
            print(f"⚠️ Failed to parse record {idx}: {e}")
            return None
    
    def _dict_to_tool_definition(self, api: Dict) -> Optional[ToolDefinition]:
        """
        将 API 字典转换为 ToolDefinition
        
        ToolBench 格式:
        {
            "name": "api_name_for_tool_name",
            "description": "...",
            "parameters": {
                "type": "object",
                "properties": {
                    "param_name": {"type": "string", "description": "...", "example_value": "..."}
                },
                "required": ["param1"],
                "optional": ["param2"]
            }
        }
        
        ToolBench 的 required/optional 判断规则：
        - 在 required 列表中 → required=True, optional=False
        - 在 optional 列表中 → required=False, optional=True
        - 都没出现 → required=False, optional=True（默认可选）
        """
        name = api.get('name', '')
        if not name:
            return None
        
        description = api.get('description', '')
        params_def = api.get('parameters', {})
        
        parameters = []
        
        if isinstance(params_def, dict):
            properties = params_def.get('properties', {})
            required_list = params_def.get('required', [])
            optional_list = params_def.get('optional', [])
            
            for param_name, param_info in properties.items():
                if not isinstance(param_info, dict):
                    continue
                
                # 判断 required/optional（互补）
                # 在 required 列表中 → required=True, optional=False
                # 在 optional 列表中 → required=False, optional=True
                # 都没出现 → required=False, optional=True（默认可选）
                in_required = param_name in required_list
                in_optional = param_name in optional_list
                
                # 四种情况显式处理，保留异常状态供 format check 检测
                if in_required and in_optional:
                    # 同时出现在两个列表中（数据异常）
                    is_required, is_optional = True, True
                elif in_required and not in_optional:
                    # 只在 required 中
                    is_required, is_optional = True, False
                elif not in_required and in_optional:
                    # 只在 optional 中
                    is_required, is_optional = False, True
                else:
                    # 都没出现，默认可选
                    is_required, is_optional = False, True
                
                # 提取 example_value 到 metadata
                metadata = {}
                if 'example_value' in param_info:
                    metadata['example_value'] = param_info['example_value']
                
                param = Parameter(
                    name=param_name,
                    type=param_info.get('type', 'string'),
                    description=param_info.get('description', ''),
                    default=None,  # ToolBench 没有 default，只有 example_value
                    required=is_required,
                    optional=is_optional,
                    metadata=metadata
                )
                parameters.append(param)
        
        return ToolDefinition(
            name=name,
            description=description,
            parameters=parameters
        )


# =============================================================================
# xLAM Loader
# =============================================================================

class XLAMLoader(BaseLoader):
    """
    xLAM-60k 数据集加载器
    
    将 xLAM 的单轮调用格式转换为统一的 APIAgentSample。
    
    xLAM 数据格式：
    {
        "query": "用户查询",
        "answers": "[{\"name\": \"api_name\", \"arguments\": {...}}, ...]",
        "tools": "[{\"name\": \"...\", \"description\": \"...\", \"parameters\": {...}}, ...]"
    }
    """
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data: List[Dict] = []
    
    def load(self) -> List[Dict]:
        """加载原始 JSONL 数据"""
        print(f"📂 Loading xLAM dataset: {self.dataset_path}")
        self.data = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"✅ Loaded {len(self.data):,} records")
        return self.data
    
    def parse_all(self, show_progress: bool = True) -> List[APIAgentSample]:
        """解析所有记录为 APIAgentSample"""
        if not self.data:
            self.load()
        
        samples = []
        iterator = tqdm(self.data, desc="Parsing xLAM") if show_progress else self.data
        
        for idx, record in enumerate(iterator):
            sample = self.parse_record(record, idx)
            if sample:
                samples.append(sample)
        
        print(f"✅ Parsed {len(samples):,} samples")
        return samples
    
    def iterate(self, show_progress: bool = True) -> Iterator[APIAgentSample]:
        """
        迭代返回 APIAgentSample（惰性加载，节省内存）
        
        Args:
            show_progress: 是否显示进度条
        
        Yields:
            APIAgentSample
        """
        if not self.data:
            self.load()
        
        iterator = tqdm(self.data, desc="Parsing xLAM") if show_progress else self.data
        
        for idx, record in enumerate(iterator):
            sample = self.parse_record(record, idx)
            if sample:
                yield sample
    
    def parse_record(self, record: Dict, idx: int = 0) -> Optional[APIAgentSample]:
        """解析单条 xLAM 记录"""
        try:
            # === 1. Query ===
            query = record.get('query', '')
            
            # === 2. 解析 Tools ===
            tools_raw = record.get('tools', '[]')
            if isinstance(tools_raw, str):
                tools_list = json.loads(tools_raw)
            else:
                tools_list = tools_raw
            
            tools = []
            for tool_data in tools_list:
                tool = self._dict_to_tool_definition(tool_data)
                if tool:
                    tools.append(tool)
            
            # === 3. 解析 Answers (API Calls) ===
            answers_raw = record.get('answers', '[]')
            if isinstance(answers_raw, str):
                answers_list = json.loads(answers_raw)
            else:
                answers_list = answers_raw
            
            api_calls = []
            for answer in answers_list:
                if isinstance(answer, dict):
                    api_call = APICall(
                        name=answer.get('name', ''),
                        arguments=answer.get('arguments', {}),
                        response=None,  # xLAM 没有响应
                        metadata={}
                    )
                    api_calls.append(api_call)
            
            # === 4. 构建 APIAgentSample ===
            sample = APIAgentSample(
                query=query,
                tools=tools,
                api_calls=api_calls,
                final_answer=None,  # xLAM 没有 final_answer
                sample_id=f"xlam_{idx}",
                source_dataset='xlam_60k',
                metadata={}
            )
            
            return sample
            
        except Exception as e:
            print(f"⚠️ Failed to parse record {idx}: {e}")
            return None
    
    def _dict_to_tool_definition(self, tool_data: Dict) -> Optional[ToolDefinition]:
        """
        解析 xLAM 工具定义
        
        xLAM 格式:
        {
            "name": "api_name",
            "description": "...",
            "parameters": {
                "param_name": {"type": "str", "description": "...", "default": "..."},
                ...
            }
        }
        
        xLAM 的 required/optional 判断规则：
        - type 中带 'optional' → required=False, optional=True
        - type 中不带 'optional' → required=True, optional=False
        """
        name = tool_data.get('name', '')
        if not name:
            return None
        
        description = tool_data.get('description', '')
        params_raw = tool_data.get('parameters', {})
        
        parameters = []
        
        if isinstance(params_raw, dict):
            for param_name, param_info in params_raw.items():
                # xLAM 格式: {param_name: {type, description, default}}
                # 每个键都是参数名，不需要跳过
                if isinstance(param_info, dict):
                    param_type = param_info.get('type', 'str')
                    
                    # 判断 required/optional（互补）
                    # type 中带 'optional' → optional=True, required=False
                    # type 中不带 'optional' → optional=False, required=True
                    if 'optional' in param_type.lower():
                        is_required, is_optional = False, True
                    else:
                        is_required, is_optional = True, False
                    
                    param = Parameter(
                        name=param_name,
                        type=param_type.split(',')[0].strip(),  # 去掉 ", optional" 部分
                        description=param_info.get('description', ''),
                        default=param_info.get('default'),
                        required=is_required,
                        optional=is_optional,
                        metadata={}
                    )
                    parameters.append(param)
        
        return ToolDefinition(
            name=name,
            description=description,
            parameters=parameters
        )


# =============================================================================
# Arcee Agent Data Loader
# =============================================================================

class ArceeAgentDataLoader(BaseLoader):
    """
    Arcee AI Agent Data 数据集加载器（过滤后的 api_only 版本）
    
    支持 5 种子集格式，根据 dataset 字段自动路由解析：
    1. glaive-function-calling-v2-extended: OpenAI function calling 格式
    2. salesforce_sharegpt: <tools>/<tool_call> 标签格式
    3. toolbench_instruct_j1s1_3k_unfiltered: 单轮 JSON 格式 {"NAME", "Args"}
    4. toolbench_react_10p_unfiltered: 多轮 JSON 格式 {"action", "Arguments"}
    5. toolbench_tflan_cot_30p_unfiltered: 多轮 Python 函数调用格式
    
    解析后自动跳过无 API 调用的样本（模型拒绝调用、纯自然语言回答等），
    跳过原因记录在 skip_stats 中。（这些拒绝样本或者自然语言样本不属于我们评测的范畴，这个数据集是训api调用agent的，而我们评测是专注于api调用能力的数据集）
    """

    _REFUSAL_PATTERNS = (
        "I'm sorry", "I don't have the ability", "I cannot",
        "I'm unable", "I can't", "unfortunately",
        "not able to", "do not have access", "beyond my capabilities",
    )
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data: List[Dict] = []
        self.skip_stats: Dict[str, Dict[str, int]] = {}
    
    def load(self) -> List[Dict]:
        """加载原始 JSONL 数据"""
        print(f"Loading Arcee Agent Data dataset: {self.dataset_path}")
        self.data = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"  Invalid JSON line: {e}")
                        continue
        print(f"  Loaded {len(self.data):,} records")
        return self.data
    
    def parse_all(self, show_progress: bool = True) -> List[APIAgentSample]:
        """解析所有记录为 APIAgentSample，自动跳过无 API 调用的样本"""
        if not self.data:
            self.load()
        
        samples = []
        self.skip_stats = {}
        iterator = tqdm(self.data, desc="Parsing Arcee Agent Data") if show_progress else self.data
        
        for idx, record in enumerate(iterator):
            sample = self.parse_record(record, idx)
            if sample:
                samples.append(sample)
        
        self._print_skip_report(len(self.data), len(samples))
        return samples
    
    def iterate(self, show_progress: bool = True) -> Iterator[APIAgentSample]:
        """迭代返回 APIAgentSample（惰性加载），自动跳过无 API 调用的样本"""
        if not self.data:
            self.load()
        
        self.skip_stats = {}
        iterator = tqdm(self.data, desc="Parsing Arcee Agent Data") if show_progress else self.data
        
        for idx, record in enumerate(iterator):
            sample = self.parse_record(record, idx)
            if sample:
                yield sample
    
    def _classify_skip_reason(self, record: Dict) -> str:
        """判断无 API 调用的原因

        Returns:
            'model_refusal'  — 模型明确拒绝调用
            'empty_response' — assistant 回复为空
            'parse_failure'  — 有调用信号（代码块/标签/JSON）但解析失败
            'no_api_call'    — 纯自然语言回答，无任何调用迹象
        """
        convs = record.get('conversations', [])
        assistant_texts = []
        for c in convs:
            if c.get('from', '').lower() in ('assistant', 'gpt'):
                assistant_texts.append(c.get('value', '').strip())
        all_text = '\n'.join(assistant_texts)

        if not all_text:
            return 'empty_response'

        first200 = all_text[:200].lower()
        if any(p.lower() in first200 for p in self._REFUSAL_PATTERNS):
            return 'model_refusal'

        if '<tool_call>' in all_text or '```' in all_text:
            return 'parse_failure'
        if all_text.strip().startswith('{') and ('"name"' in all_text or '"action"' in all_text):
            return 'parse_failure'

        return 'no_api_call'

    def _record_skip(self, dataset: str, reason: str):
        if dataset not in self.skip_stats:
            self.skip_stats[dataset] = {}
        self.skip_stats[dataset][reason] = self.skip_stats[dataset].get(reason, 0) + 1

    _EXPECTED_REASONS = ('model_refusal', 'no_api_call')
    _DATA_ISSUE_REASONS = ('empty_response', 'parse_failure')

    def _print_skip_report(self, total: int, kept: int):
        skipped = total - kept
        print(f"  Parsed {kept:,} samples, skipped {skipped:,}")
        if not self.skip_stats:
            return

        expected_total = 0
        issue_total = 0
        expected_details = []
        issue_details = []

        for ds in sorted(self.skip_stats):
            for reason, cnt in sorted(self.skip_stats[ds].items()):
                entry = f"    {ds}: {reason} ({cnt:,})"
                if reason in self._EXPECTED_REASONS:
                    expected_total += cnt
                    expected_details.append(entry)
                else:
                    issue_total += cnt
                    issue_details.append(entry)

        print(f"  Expected skips (model refusal / pure text answer): {expected_total:,}")
        for line in expected_details:
            print(line)
        print(f"  Data quality issues (incomplete data / parse failure): {issue_total:,}")
        for line in issue_details:
            print(line)

    def parse_record(self, record: Dict, idx: int = 0) -> Optional[APIAgentSample]:
        """根据 dataset 字段自动路由解析，无 API 调用的样本返回 None"""
        dataset = record.get('dataset', '')
        try:
            if dataset == 'glaive-function-calling-v2-extended':
                sample = self._parse_glaive(record, idx)
            elif dataset == 'salesforce_sharegpt':
                sample = self._parse_salesforce(record, idx)
            elif dataset.startswith('toolbench_'):
                sample = self._parse_toolbench(record, idx)
            else:
                return None
        except Exception as e:
            print(f"  Failed to parse record {idx} ({dataset}): {e}")
            return None

        if sample and not sample.api_calls:
            reason = self._classify_skip_reason(record)
            self._record_skip(dataset, reason)
            return None
        return sample

    # =========================================================================
    # glaive-function-calling-v2-extended
    # =========================================================================
    # system: OpenAI function calling JSON 数组
    #   [{"type":"function","function":{"name":"...","parameters":{"properties":{...},"required":[...]}}}]
    # gpt 回复: JSON 数组 [{"name":"...","arguments":{...}}]

    def _parse_glaive(self, record: Dict, idx: int) -> Optional[APIAgentSample]:
        conversations = record.get('conversations', [])
        system_prompt = ""
        query = ""
        api_calls = []
        assistant_texts = []

        for conv in conversations:
            role = conv.get('from', '').lower()
            value = conv.get('value', '').strip()
            if role == 'system':
                system_prompt = value
            elif role == 'human':
                if not query:
                    query = value
            elif role in ('gpt', 'assistant'):
                assistant_texts.append(value)

        tools = self._parse_glaive_tools(system_prompt)

        for text in assistant_texts:
            api_calls.extend(self._parse_glaive_calls(text))

        final_answer = assistant_texts[-1] if assistant_texts else ""

        return APIAgentSample(
            query=query,
            tools=tools,
            api_calls=api_calls,
            final_answer=final_answer,
            sample_id=f"arcee_glaive_{idx}",
            source_dataset='glaive-function-calling-v2-extended',
            metadata={
                'raw_conversations': conversations,
                'system_prompt': system_prompt,
            }
        )

    def _parse_glaive_tools(self, system_prompt: str) -> List[ToolDefinition]:
        """从 system prompt 中解析 OpenAI function calling 格式的工具定义"""
        tools = []
        bracket_start = system_prompt.find('[')
        if bracket_start == -1:
            return tools

        depth = 0
        for i in range(bracket_start, len(system_prompt)):
            if system_prompt[i] == '[':
                depth += 1
            elif system_prompt[i] == ']':
                depth -= 1
                if depth == 0:
                    json_str = system_prompt[bracket_start:i + 1]
                    break
        else:
            return tools

        try:
            tool_list = json.loads(json_str)
        except json.JSONDecodeError:
            return tools

        if not isinstance(tool_list, list):
            return tools

        for item in tool_list:
            if not isinstance(item, dict):
                continue
            func_def = item.get('function', item)
            if not isinstance(func_def, dict):
                continue
            name = func_def.get('name', '').strip()
            if not name:
                continue
            description = func_def.get('description', '').strip()
            params_schema = func_def.get('parameters') or {}
            parameters = self._parse_openai_params(params_schema)
            tools.append(ToolDefinition(name=name, description=description, parameters=parameters))

        return tools

    def _parse_openai_params(self, params_schema: Dict) -> List[Parameter]:
        """解析 OpenAI 格式的 parameters（只有 required 列表）"""
        parameters = []
        if not isinstance(params_schema, dict):
            return parameters

        properties = params_schema.get('properties', {})
        if not isinstance(properties, dict):
            return parameters

        required_raw = params_schema.get('required', [])
        required_list = required_raw if isinstance(required_raw, list) else []

        for param_name, param_info in properties.items():
            if not isinstance(param_info, dict):
                continue
            # 有些数据在参数级别也有 "required": true/false
            param_required = param_info.get('required')
            if isinstance(param_required, bool):
                in_required = param_required
            else:
                in_required = param_name in required_list
            parameters.append(Parameter(
                name=param_name,
                type=param_info.get('type', 'string'),
                description=param_info.get('description', ''),
                default=None,
                required=in_required,
                optional=not in_required,
                metadata={}
            ))
        return parameters

    def _parse_glaive_calls(self, gpt_text: str) -> List[APICall]:
        """从 gpt 回复中解析 JSON 数组格式的函数调用"""
        calls = []
        bracket_start = gpt_text.find('[')
        if bracket_start == -1:
            return calls

        depth = 0
        for i in range(bracket_start, len(gpt_text)):
            if gpt_text[i] == '[':
                depth += 1
            elif gpt_text[i] == ']':
                depth -= 1
                if depth == 0:
                    json_str = gpt_text[bracket_start:i + 1]
                    break
        else:
            return calls

        try:
            call_list = json.loads(json_str)
        except json.JSONDecodeError:
            return calls

        if not isinstance(call_list, list):
            return calls

        for ci, item in enumerate(call_list):
            if isinstance(item, dict) and 'name' in item:
                calls.append(APICall(
                    name=item.get('name', ''),
                    arguments=item.get('arguments', {}),
                    response=None,
                    metadata={'tool_call_index': ci}
                ))
        return calls

    # =========================================================================
    # salesforce_sharegpt
    # =========================================================================
    # system: <tools>[...]</tools> 标签
    # gpt 回复: <tool_call>{...}</tool_call> 标签

    def _parse_salesforce(self, record: Dict, idx: int) -> Optional[APIAgentSample]:
        conversations = record.get('conversations', [])
        system_prompt = ""
        query = ""
        final_answer = ""

        for conv in conversations:
            role = conv.get('from', '').lower()
            value = conv.get('value', '').strip()
            if role == 'system':
                system_prompt = value
            elif role == 'human':
                query = value
            elif role in ('gpt', 'assistant'):
                final_answer = value

        tools = self._parse_salesforce_tools(system_prompt)
        api_calls = self._parse_salesforce_calls(final_answer)

        return APIAgentSample(
            query=query,
            tools=tools,
            api_calls=api_calls,
            final_answer=final_answer,
            sample_id=f"arcee_salesforce_{idx}",
            source_dataset='salesforce_sharegpt',
            metadata={
                'raw_conversations': conversations,
                'system_prompt': system_prompt,
            }
        )

    def _parse_salesforce_tools(self, system_prompt: str) -> List[ToolDefinition]:
        """从 <tools> 标签中解析工具定义"""
        tools = []
        tool_pattern = re.compile(r'<tools>(.*?)</tools>', re.DOTALL)
        tool_matches = tool_pattern.findall(system_prompt)
        if not tool_matches:
            return tools

        try:
            tool_list = json.loads(tool_matches[0])
            if not isinstance(tool_list, list):
                return tools
            for tool_data in tool_list:
                tool_def = self._salesforce_dict_to_tool(tool_data)
                if tool_def:
                    tools.append(tool_def)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return tools

    def _salesforce_dict_to_tool(self, tool_data: Dict) -> Optional[ToolDefinition]:
        """salesforce 格式：参数是扁平 dict，靠 default 判断 required"""
        name = tool_data.get('name', '').strip()
        if not name:
            return None

        description = tool_data.get('description', '').strip()
        params_raw = tool_data.get('parameters', {})
        parameters = []

        if isinstance(params_raw, dict):
            for param_name, param_info in params_raw.items():
                if not isinstance(param_info, dict):
                    continue
                param_default = param_info.get('default')
                is_required = param_default is None
                parameters.append(Parameter(
                    name=param_name,
                    type=param_info.get('type', 'str'),
                    description=param_info.get('description', '').strip(),
                    default=param_default,
                    required=is_required,
                    optional=not is_required,
                    metadata={}
                ))
        return ToolDefinition(name=name, description=description, parameters=parameters)

    def _parse_salesforce_calls(self, gpt_response: str) -> List[APICall]:
        """从 <tool_call> 标签中解析 API 调用"""
        api_calls = []
        call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
        call_matches = call_pattern.findall(gpt_response)
        if not call_matches:
            return api_calls

        for ci, call_str in enumerate(call_matches):
            call_data = None
            stripped = call_str.strip()
            try:
                call_data = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                try:
                    call_data = ast.literal_eval(stripped)
                except (ValueError, SyntaxError):
                    pass
            if isinstance(call_data, dict) and call_data.get('tool_name'):
                api_calls.append(APICall(
                    name=call_data['tool_name'],
                    arguments=call_data.get('tool_arguments', {}),
                    response=None,
                    metadata={'tool_call_index': ci}
                ))
        return api_calls

    # =========================================================================
    # toolbench_* (instruct / react / tflan_cot)
    # =========================================================================
    # 工具定义格式统一：文本中嵌 Python dict 或 JSON，有 required + optional 列表
    # API 调用格式各异：
    #   instruct: {"NAME": ..., "Args": {...}}
    #   react:    {"action": ..., "Arguments": {...}}
    #   tflan_cot: Python 函数调用 api_name(args) 或自然语言

    def _parse_toolbench(self, record: Dict, idx: int) -> Optional[APIAgentSample]:
        conversations = record.get('conversations', [])
        dataset = record.get('dataset', '')

        system_prompt = ""
        query = ""
        api_calls = []
        final_answer = None

        for conv in conversations:
            role = conv.get('from', '').lower()
            value = conv.get('value', '').strip()
            if role == 'system':
                system_prompt = value
            elif role in ('human', 'user'):
                if not query:
                    query = value

        tools = self._parse_toolbench_tools(system_prompt)
        template_keys = self._extract_template_keys(system_prompt)

        # 遍历多轮对话，配对 assistant 和 user(function response)
        i = 0
        while i < len(conversations):
            conv = conversations[i]
            role = conv.get('from', '').lower()
            value = conv.get('value', '').strip()

            if role in ('assistant', 'gpt'):
                action_name, action_args, thought = self._parse_toolbench_action(
                    value, dataset, template_keys)

                response = None
                if i + 1 < len(conversations):
                    next_role = conversations[i + 1].get('from', '').lower()
                    if next_role in ('user', 'human', 'function'):
                        next_val = conversations[i + 1].get('value', '').strip()
                        if next_val.startswith('{') and 'response' in next_val:
                            response = next_val

                if action_name:
                    if action_name == 'FinalAction':
                        if isinstance(action_args, dict):
                            final_answer = action_args.get('final_answer', '')
                    api_calls.append(APICall(
                        name=action_name,
                        arguments=action_args if isinstance(action_args, dict) else {},
                        response=response,
                        metadata={'thought': thought} if thought else {}
                    ))
                elif not action_name and not api_calls:
                    pass
                else:
                    final_answer = value
            i += 1

        if final_answer is None and api_calls:
            last_call = api_calls[-1]
            if last_call.name == 'FinalAction':
                final_answer = last_call.arguments.get('final_answer', '')

        return APIAgentSample(
            query=query,
            tools=tools,
            api_calls=api_calls,
            final_answer=final_answer,
            sample_id=f"arcee_toolbench_{idx}",
            source_dataset=dataset,
            metadata={
                'raw_conversations': conversations,
                'system_prompt': system_prompt,
            }
        )

    def _parse_toolbench_tools(self, system_prompt: str) -> List[ToolDefinition]:
        """从 toolbench system prompt 中解析工具定义"""
        tools = []

        # 格式1: instruct 单个 Python dict
        if "You have access to the following API:" in system_prompt:
            api_start = system_prompt.find("You have access to the following API:")
            api_text = system_prompt[api_start + len("You have access to the following API:"):].strip()
            brace_start = api_text.find('{')
            if brace_start != -1:
                try:
                    api_dict = ast.literal_eval(api_text[brace_start:].split('\n')[0].strip())
                    if isinstance(api_dict, dict) and api_dict.get('name'):
                        tools.append(ToolDefinition(
                            name=api_dict['name'],
                            description=api_dict.get('description', ''),
                            parameters=self._parse_toolbench_params(api_dict.get('parameters', {}))
                        ))
                        return tools
                except (ValueError, SyntaxError):
                    pass

        # 格式2: react/tflan_cot 逐段 "api_name: description\nInput parameters..."
        lines = system_prompt.split('\n')
        current_name = None
        current_desc = ""

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith('Input parameters are as follows:'):
                params_part = line_stripped[len('Input parameters are as follows:'):].strip()
                if params_part and current_name:
                    try:
                        params_dict = json.loads(params_part)
                        tools.append(ToolDefinition(
                            name=current_name,
                            description=current_desc,
                            parameters=self._parse_toolbench_params(params_dict)
                        ))
                    except json.JSONDecodeError:
                        pass
                    current_name = None
                    current_desc = ""
                continue

            if ': ' in line_stripped and not line_stripped.startswith('{') and not line_stripped.startswith('"'):
                parts = line_stripped.split(': ', 1)
                candidate_name = parts[0].strip()
                if candidate_name and not candidate_name.startswith('#') and not candidate_name.startswith('//'):
                    if re.match(r'^[a-zA-Z_][\w.]*$', candidate_name) or candidate_name == 'FinalAction':
                        current_name = candidate_name
                        current_desc = parts[1].strip() if len(parts) > 1 else ""

        return tools

    def _parse_toolbench_params(self, params_schema: Dict) -> List[Parameter]:
        """解析 toolbench 格式的参数（有 required + optional 列表）"""
        parameters = []
        properties = params_schema.get('properties', {})
        required_list = params_schema.get('required', [])
        optional_list = params_schema.get('optional', [])

        for param_name, param_info in properties.items():
            if not isinstance(param_info, dict):
                continue

            in_required = param_name in required_list
            in_optional = param_name in optional_list

            if in_required and in_optional:
                is_required, is_optional = True, True
            elif in_required:
                is_required, is_optional = True, False
            elif in_optional:
                is_required, is_optional = False, True
            else:
                is_required, is_optional = False, True

            metadata = {}
            if 'example_value' in param_info:
                metadata['example_value'] = param_info['example_value']

            parameters.append(Parameter(
                name=param_name,
                type=param_info.get('type', 'string'),
                description=param_info.get('description', ''),
                default=None,
                required=is_required,
                optional=is_optional,
                metadata=metadata
            ))
        return parameters

    def _parse_toolbench_action(self, assistant_text: str,
                                 dataset: str,
                                 template_keys: Tuple = (None, None, None, 'unknown')
                                 ) -> Tuple[Optional[str], Optional[Dict], str]:
        """从 assistant 回复中解析 API 调用，返回 (action_name, arguments, thought)

        template_keys 由调用方提前从 system prompt 中提取一次（避免每轮重复提取）。
        """
        thought_key, action_key, args_key, fmt = template_keys

        # JSON 模板解析（instruct / react / tflan_cot 共用）
        if fmt == 'json' and action_key:
            try:
                data = json.loads(assistant_text)
                action_name = data.get(action_key, '')
                if action_name:
                    args = data.get(args_key, {}) if args_key else {}
                    thought = data.get(thought_key, '') if thought_key else ''
                    return (action_name,
                            args if isinstance(args, dict) else {},
                            thought)
            except json.JSONDecodeError:
                pass
            if dataset != 'toolbench_tflan_cot_30p_unfiltered':
                return (None, None, "")

        # tflan_cot: 多种代码块格式（```python/json/yaml/bash）
        if dataset == 'toolbench_tflan_cot_30p_unfiltered':
            result = self._parse_tflan_code_block(assistant_text)
            if result[0]:
                return result

        # 纯文本模板解析
        if fmt == 'text' and action_key:
            return self._parse_text_action(
                assistant_text, thought_key, action_key, args_key)

        return (None, None, "")

    def _parse_tflan_code_block(self, assistant_text: str
                                 ) -> Tuple[Optional[str], Optional[Dict], str]:
        """解析 tflan_cot 中各种代码块格式的函数调用。

        支持: ```python, ```json, ```yaml, ```bash, ``` (空标记)
        """
        thought = ""
        thought_match = re.match(r'(.*?)(?:I will|```)', assistant_text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        code_match = re.search(r'```(\w*)\s*\n(.*?)\n```', assistant_text, re.DOTALL)
        if not code_match:
            return (None, None, "")

        tag = code_match.group(1).lower()
        content = code_match.group(2).strip()

        # python/typescript: func_name(key=val, ...) 或 {'name': '...', 'parameters': {...}}
        if tag in ('python', 'typescript'):
            func_match = re.match(r'([\w.]+)\((.*)\)', content, re.DOTALL)
            if func_match:
                func_name = func_match.group(1)
                args_str = func_match.group(2).strip()
                args = {}
                if args_str:
                    for part in re.findall(r'(\w+)\s*=\s*("[^"]*"|\S+)', args_str):
                        args[part[0]] = part[1].strip('"').strip("'")
                return (func_name, args, thought)
            # Python dict 格式: {'name': 'api_name', 'parameters': {...}}
            if content.startswith('{'):
                try:
                    data = ast.literal_eval(content)
                    if isinstance(data, dict) and 'name' in data:
                        params = data.get('parameters', {})
                        return (data['name'],
                                params if isinstance(params, dict) else {},
                                thought)
                except (ValueError, SyntaxError):
                    pass

        # json: {"name": "api_name", "parameters": {...}}
        if tag == 'json' or (tag == '' and content.startswith('{')):
            try:
                data = json.loads(content)
                if isinstance(data, dict) and 'name' in data:
                    params = data.get('parameters', {})
                    return (data['name'],
                            params if isinstance(params, dict) else {},
                            thought)
            except json.JSONDecodeError:
                pass

        # yaml: name: api_name\nparameters:\n  key: val
        if tag == 'yaml' or (tag == '' and content.startswith('name:')):
            lines = content.split('\n')
            func_name = ""
            args = {}
            in_params = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('name:'):
                    func_name = stripped[len('name:'):].strip()
                elif stripped == 'parameters:':
                    in_params = True
                elif in_params and ':' in stripped:
                    k, v = stripped.split(':', 1)
                    args[k.strip()] = v.strip()
                elif not stripped.startswith(' ') and not stripped.startswith('\t'):
                    in_params = False
            if func_name:
                return (func_name, args, thought)

        # bash: api_name --key value --key2 value2
        if tag == 'bash':
            parts = content.split()
            if parts:
                func_name = parts[0]
                args = {}
                i = 1
                while i < len(parts):
                    if parts[i].startswith('--') and i + 1 < len(parts):
                        args[parts[i][2:]] = parts[i + 1]
                        i += 2
                    else:
                        i += 1
                return (func_name, args, thought)

        # plaintext: 尝试提取 "api_name" 或类似模式
        if tag == 'plaintext':
            func_match = re.match(r'([\w.]+(?:_for_\w+)?)\s*(.*)', content)
            if func_match:
                func_name = func_match.group(1)
                rest = func_match.group(2).strip()
                args = {}
                if rest:
                    for pair in re.findall(r'(\w+)\s*[:=]\s*("[^"]*"|\S+)', rest):
                        args[pair[0]] = pair[1].strip('"')
                if '.' in func_name or '_for_' in func_name:
                    return (func_name, args, thought)

        return (None, None, "")

    # =========================================================================
    # 模板 key 提取
    # =========================================================================

    _THOUGHT_HINTS = ('thought', 'cot', 'think', 'reason', 'goal', 'inner')
    _ACTION_HINTS = ('api', 'name', 'tool', 'action', 'call', 'command')
    _ARGS_HINTS = ('param', 'arg', 'input', 'json')

    def _extract_template_keys(self, system_prompt: str
                                ) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
        """从 system prompt 中提取模板 key 名 (thought_key, action_key, args_key, format_type)。

        优先从 ``` 代码块中提取 JSON 模板 key，失败则回退到文本模板。
        """
        # 尝试 JSON 模板（``` 代码块）
        code_match = re.search(r'```\s*\n(.*?)\n```', system_prompt, re.DOTALL)
        if code_match:
            thought_key, action_key, args_key = None, None, None
            prev_comment = ""
            for line in code_match.group(1).split('\n'):
                stripped = line.strip()
                if stripped.startswith('//'):
                    prev_comment = stripped[2:].strip().lower()
                    continue
                key_match = re.match(r'"(\w+)"\s*:', stripped)
                if not key_match:
                    prev_comment = ""
                    continue
                key_name = key_match.group(1)
                rest = stripped[key_match.end():].strip().rstrip(';').strip()
                if 'Record' in rest or 'object' in rest.lower():
                    args_key = key_name
                elif any(h in prev_comment for h in self._THOUGHT_HINTS):
                    thought_key = key_name
                elif any(h in prev_comment for h in self._ACTION_HINTS):
                    action_key = key_name
                else:
                    if thought_key is None:
                        thought_key = key_name
                    elif action_key is None:
                        action_key = key_name
                prev_comment = ""
            if action_key:
                return (thought_key, action_key, args_key, 'json')

        # 回退：纯文本模板
        thought_key, action_key, args_key = None, None, None
        format_start = -1
        for marker in ('following format:', 'Follow the format', 'the format, i.e'):
            idx = system_prompt.find(marker)
            if idx != -1:
                format_start = idx + len(marker)
                break
        if format_start == -1:
            return (None, None, None, 'unknown')

        for line in system_prompt[format_start:format_start + 500].split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            candidate = stripped.split(':')[0].strip() if ':' in stripped else (stripped.split()[0] if stripped.split() else '')
            if not candidate or len(candidate) > 30:
                continue
            desc = stripped.lower()
            if thought_key is None and any(h in desc for h in self._THOUGHT_HINTS):
                thought_key = candidate
            elif action_key is None and any(h in desc for h in self._ACTION_HINTS):
                action_key = candidate
            elif args_key is None and any(h in desc for h in self._ARGS_HINTS):
                args_key = candidate

        return (thought_key, action_key, args_key, 'text')

    def _parse_text_action(self, assistant_text: str,
                            thought_key: Optional[str],
                            action_key: str,
                            args_key: Optional[str]
                            ) -> Tuple[Optional[str], Optional[Dict], str]:
        """解析纯文本格式的 assistant 回复。

        根据提取到的 key 名，在 assistant 文本中按行匹配。
        """
        thought = ""
        action_name = ""
        args = {}

        keys = []
        if thought_key:
            keys.append((thought_key, 'thought'))
        keys.append((action_key, 'action'))
        if args_key:
            keys.append((args_key, 'args'))
        keys.sort(key=lambda x: len(x[0]), reverse=True)

        for line in assistant_text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue

            for key, role in keys:
                if not stripped.startswith(key):
                    continue
                rest = stripped[len(key):].strip()
                if rest.startswith(':'):
                    rest = rest[1:].strip()

                if role == 'thought':
                    thought = rest
                elif role == 'action':
                    action_name = rest
                elif role == 'args':
                    try:
                        args = json.loads(rest)
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                break

        if action_name:
            return (action_name, args if isinstance(args, dict) else {}, thought)
        return (None, None, "")


# =============================================================================
# 便捷函数
# =============================================================================

def load_toolbench(path: str, show_progress: bool = True) -> List[APIAgentSample]:
    """便捷函数：加载 ToolBench 数据集"""
    loader = ToolBenchLoader(path)
    return loader.parse_all(show_progress)


def load_xlam(path: str, show_progress: bool = True) -> List[APIAgentSample]:
    """便捷函数：加载 xLAM-60k 数据集"""
    loader = XLAMLoader(path)
    return loader.parse_all(show_progress)


def load_arcee_agent_data(path: str, show_progress: bool = True) -> List[APIAgentSample]:
    """便捷函数：加载 Arcee AI Agent Data 数据集"""
    loader = ArceeAgentDataLoader(path)
    return loader.parse_all(show_progress)


# =============================================================================
# 测试
# =============================================================================

def print_sample(sample: APIAgentSample):
    """打印完整的 APIAgentSample 结构"""
    print("=" * 80)
    print(f"📋 APIAgentSample")
    print("=" * 80)
    
    # 基本信息
    print(f"\n📌 sample_id: {sample.sample_id}")
    print(f"📌 source_dataset: {sample.source_dataset}")
    print(f"📌 metadata: {sample.metadata}")
    
    # Query
    print(f"\n📝 Query:")
    print(f"   {sample.query}")
    
    # Final Answer
    if sample.final_answer:
        print(f"\n✅ Final Answer:")
        print(f"   {sample.final_answer[:200]}..." if len(sample.final_answer) > 200 else f"   {sample.final_answer}")
    
    # Tools
    print(f"\n🛠️  Tools ({len(sample.tools)}):")
    for i, tool in enumerate(sample.tools):
        print(f"\n   [{i}] ToolDefinition:")
        print(f"       name: {tool.name}")
        print(f"       description: {tool.description[:100]}..." if len(tool.description) > 100 else f"       description: {tool.description}")
        print(f"       parameters ({len(tool.parameters)}):")
        for p in tool.parameters:
            print(f"         - Parameter:")
            print(f"             name: {p.name}")
            print(f"             type: {p.type}")
            print(f"             description: {p.description[:80]}..." if len(p.description) > 80 else f"             description: {p.description}")
            print(f"             default: {p.default}")
            print(f"             required: {p.required}")
            print(f"             optional: {p.optional}")
            if p.metadata:
                print(f"             metadata: {p.metadata}")
    
    # API Calls
    print(f"\n📞 API Calls ({len(sample.api_calls)}):")
    for i, call in enumerate(sample.api_calls):
        print(f"\n   [{i}] APICall:")
        print(f"       name: {call.name}")
        print(f"       arguments: {call.arguments}")
        if call.response is not None:
            resp_str = str(call.response)
            print(f"       response: {resp_str[:150]}..." if len(resp_str) > 150 else f"       response: {resp_str}")
        if call.metadata:
            thought = call.metadata.get('thought', '')
            if thought:
                print(f"       metadata.thought: {thought[:150]}..." if len(thought) > 150 else f"       metadata.thought: {thought}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    # 测试 ToolBench Loader
    print("\n" + "=" * 80)
    print("Testing ToolBench Loader")
    print("=" * 80)
    
    toolbench_path = '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/toolbench_official/toolllama_G123_dfs_train.json'
    loader = ToolBenchLoader(toolbench_path)
    loader.load()
    
    sample = loader.parse_record(loader.data[0], 0)
    if sample:
        print_sample(sample)
    
    print("\n" + "=" * 80)
    print("Testing xLAM Loader")
    print("=" * 80)
    
    xlam_path = '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/xlam_60k.jsonl'
    loader = XLAMLoader(xlam_path)
    loader.load()
    
    sample = loader.parse_record(loader.data[0], 0)
    if sample:
        print_sample(sample)

    # 测试 Arcee Agent Data Loader
    # print("\n" + "=" * 80)
    # print("Testing Arcee Agent Data Loader")
    # print("=" * 80)
    # arcee_path = '/path/to/arcee_agent_data.jsonl'
    # loader = ArceeAgentDataLoader(arcee_path)
    # loader.load()
    # if loader.data:
    #     sample = loader.parse_record(loader.data[0], 0)
    #     if sample:
    #         print_sample(sample)
