#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Agent Data Loaders

数据加载器，将不同格式的数据集转换为统一的 APIAgentSample 格式。

支持：
- ToolBench: 多轮对话格式，有真实 API 响应
- xLAM-60k: 单轮调用格式，只有工具定义

解析函数复用自 evaluate_toolbench_basic.py（经过验证的代码）。
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
# 工具函数（复用自 evaluate_toolbench_basic.py）
# =============================================================================

def extract_balanced_braces(text: str) -> str:
    """
    提取平衡的大括号内容
    
    处理嵌套大括号和字符串内的大括号，确保提取完整的 JSON 对象
    
    复用自 evaluate_toolbench_basic.py._extract_balanced_braces
    """
    if not text or text[0] != '{':
        return ""
    
    depth = 0  # 大括号嵌套深度
    in_string = False  # 当前是否在字符串内
    escape = False  # 前一个字符是否是转义符 
    # {"message": "He said \"Hello\""}
    
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
    
    # 不完整，返回全部（让后续解析器处理错误）
    return text


def fix_json_newlines(json_str: str) -> str:
    """
    处理嵌套 JSON 解析时产生的换行符问题
    
    【背景】
    ToolBench 数据结构是嵌套 JSON：
      原始文件: {"value": "Action Input: {\"answer\": \"Line1\\nLine2\"}"}
                                                          ^^
                                                文件中是转义序列 \n
    
    【问题】
    当 json.load() 解析外层 JSON 时：
      value = 'Action Input: {"answer": "Line1\nLine2"}'
                                              ^
                                    变成了真正的换行符 (ASCII 10)
    
    此时如果对内层 JSON 再 json.loads()，会失败：
      json.loads('{"answer": "Line1\nLine2"}')  # ❌ Invalid control character
    
    【解决方案】
    把字符串值内的真正换行符转回 \n 转义序列：
      '{"answer": "Line1\nLine2"}'  →  '{"answer": "Line1\\nLine2"}'
                       ^                              ^^
                 真正换行符                        转义序列
    
    【注意】
    只转换字符串值内的换行符，不转换 JSON 结构中的换行符：
      {"a": "line1\nline2"}  中字符串内的 \n 需要转换 ✅
      {\n"a": "b"}          中结构的 \n 不需要转换 ❌
    
    【验证】
    - 如果原数据没问题：直接解析成功，fix 后结果相同
    - 如果有嵌套解析问题：直接解析失败，fix 后成功（不是数据问题）
    - 如果是真正的数据问题：fix 后仍然失败（报告为数据质量问题）
    
    复用自 evaluate_toolbench_basic.py._fix_json_newlines
    """
    result = []
    in_string = False
    i = 0
    
    while i < len(json_str):
        char = json_str[i]
        
        # 1. 处理转义字符 \x
        if char == '\\' and i + 1 < len(json_str):
            result.append(char)
            result.append(json_str[i + 1])
            i += 2
            continue
        
        # 2. 处理引号 "
        if char == '"':
            in_string = not in_string
            result.append(char)
            i += 1
            continue
        
        # 3. 处理其他字符
        # 在字符串内，将实际换行符替换为 \n
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


def is_valid_api_name(name: str) -> bool:
    """
    判断是否是有效的 API 名
    
    复用自 evaluate_toolbench_basic.py._parse_action 中的内部函数
    """
    if not name:
        return False
    # 过滤纯数字+标点（如 1., 2., 3.）
    if re.match(r'^\d+\.?$', name):
        return False
    # 过滤常见的描述性词汇（如 Call, Use, Invoke 等）
    if name.lower() in ['call', 'use', 'invoke', 'execute', 'run', 'the', 'a', 'an', 'to', 'for', 'with']:
        return False
    # 过滤纯符号（如 -, *, >, ##, **）
    if re.match(r'^[\-\*\>\#\.\,\!\?\:\;\(\)\[\]\{\}]+$', name):
        return False
    # 过滤太短且不包含下划线的（如 I, A, The）
    if len(name) <= 2 and '_' not in name:
        return False
    # 必须包含至少一个字母
    if not re.search(r'[a-zA-Z]', name):
        return False
    return True


def parse_system_apis(system_text: str) -> List[Dict]:
    """
    从 system prompt 中解析 API 定义
    
    ToolBench 的 API 定义格式：
    Specifically, you have access to the following APIs: [{...}, {...}, ...]
    
    复用自 evaluate_toolbench_basic.py._parse_system_apis
    """
    apis = []
    
    # 找到 API 列表
    marker = "Specifically, you have access to the following APIs:"
    start = system_text.find(marker)
    if start == -1:
        return apis
    
    api_text = system_text[start + len(marker):].strip()
    
    try:
        # 使用 ast.literal_eval 解析 Python 格式的列表
        apis = ast.literal_eval(api_text)
    except:
        # 尝试 JSON 解析
        try:
            apis = json.loads(api_text)
        except:
            pass
    
    return apis if isinstance(apis, list) else []


def parse_action(assistant_text: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    从 assistant 回复中解析 Action 和 Action Input
    
    格式：
    Thought: ...
    Action: api_name
    Action Input: {"param": "value"}
    
    注意：某些数据中 LLM 可能先写描述性的 Action:，后面才是真正的 API 调用
    例如：
        Action: 1. Call the "some_api" function...
        ...
        Action: some_api_for_tool
        Action Input: {...}
    
    策略：
    1. 找所有 Action: 匹配
    2. 过滤明显无效的 API 名（如数字+标点、纯符号等）
    3. 优先选择有效的 API 名
    
    复用自 evaluate_toolbench_basic.py._parse_action
    """
    action_name = None
    action_input = None
    
    # 解析 Action - 使用 findall 找所有匹配，然后过滤选择最佳
    action_matches = re.findall(r'Action:\s*(\S+)', assistant_text)
    
    # 选择最后一个有效的 API 名（从后往前找）
    # 因为 LLM 有时会先写描述性的 Action:，真正的 API 调用在后面
    for match in reversed(action_matches):
        candidate = match.strip()
        if is_valid_api_name(candidate):
            action_name = candidate
            break
    
    # 如果没有找到有效的，使用最后一个匹配作为 fallback
    if action_name is None and action_matches:
        action_name = action_matches[-1].strip()
    
    # 解析 Action Input - 使用智能括号匹配
    input_start = assistant_text.find('Action Input:')
    if input_start != -1:
        # 找到第一个 {
        brace_start = assistant_text.find('{', input_start)
        if brace_start != -1:
            # 使用智能括号匹配提取完整的 JSON 对象
            input_str = extract_balanced_braces(assistant_text[brace_start:])
            if input_str:
                # 处理嵌套 JSON 解析问题
                input_str_fixed = fix_json_newlines(input_str)
                
                try:
                    action_input_direct = json.loads(input_str)
                    # 直接解析成功，验证 fix 后结果一致
                    action_input_fixed = json.loads(input_str_fixed)
                    assert action_input_direct == action_input_fixed, \
                        f"fix_json_newlines 改变了数据内容！"
                    action_input = action_input_direct
                except json.JSONDecodeError:
                    # 直接解析失败，尝试 fix 后解析
                    try:
                        action_input = json.loads(input_str_fixed)
                    except:
                        # fix 后仍失败，尝试 ast.literal_eval
                        try:
                            action_input = ast.literal_eval(input_str)
                        except:
                            pass
    
    return action_name, action_input


def extract_thought(assistant_text: str) -> str:
    """
    从 assistant 回复中提取 Thought 部分
    
    复用自 evaluate_toolbench_basic.py._extract_thought
    """
    # Thought: ... Action: 之间的内容
    match = re.search(r'Thought:\s*(.*?)(?:Action:|$)', assistant_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_step_number(id_str: str) -> Optional[int]:
    """
    从 id 中提取 Step 数字
    
    复用自 evaluate_toolbench_basic.py._extract_step_number
    """
    if id_str.startswith('Step '):
        match = re.match(r'Step (\d+):', id_str)
        if match:
            return int(match.group(1))
    return None


def extract_query(id_str: str) -> str:
    """
    从 id 中提取用户指令
    
    复用自 evaluate_toolbench_basic.py._extract_query
    """
    if ':' in id_str:
        return ':'.join(id_str.split(':')[1:]).strip()
    return id_str


def parse_function_response(func_value: str) -> Optional[str]:
    """
    解析 function 响应
    
    训练数据格式: {"error": "...", "response": "..."}
    
    由于 ToolBench 数据中 response 字段可能是 Python repr 格式（单引号），
    标准 JSON 解析会失败。直接返回原始字符串。
    
    判断是否失败时，用关键词搜索（timeout, exception, failed 等）。
    
    Returns:
        原始 function value 字符串
    """
    return func_value if func_value else None


# =============================================================================
# ToolBench Loader
# =============================================================================

class ToolBenchLoader:
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
        
        使用复用自 evaluate_toolbench_basic.py 的解析函数.
        
        Args:
            record: 原始记录
            idx: 记录索引
        
        Returns:
            APIAgentSample 或 None（解析失败时）
        """
        try:
            # === 1. 解析 ID 和 Query ===
            id_str = record.get('id', '')
            step_number = extract_step_number(id_str)
            query = extract_query(id_str)
            
            conversations = record.get('conversations', [])
            if not conversations:
                return None
            
            # === 2. 从 system prompt 解析工具定义 ===
            tools = []
            for conv in conversations:
                if conv.get('from') == 'system':
                    api_dicts = parse_system_apis(conv.get('value', ''))
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
                    
                    action_name, action_input = parse_action(assistant_text)
                    thought = extract_thought(assistant_text)
                    
                    if action_name:
                        # 查找下一个 function 响应
                        response = None
                        
                        if i + 1 < len(conversations) and conversations[i + 1].get('from') == 'function':
                            func_value = conversations[i + 1].get('value', '')
                            response = parse_function_response(func_value)
                        
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
                
                # 判断是否可选：忠实于源数据
                # 在 optional 列表中 → True
                # 在 required 列表中 → False
                # 两者都不在 → False（默认必需）
                is_optional = param_name in optional_list
                
                # 提取 example_value 到 metadata
                metadata = {}
                if 'example_value' in param_info:
                    metadata['example_value'] = param_info['example_value']
                
                param = Parameter(
                    name=param_name,
                    type=param_info.get('type', 'string'),
                    description=param_info.get('description', ''),
                    default=None,  # ToolBench 没有 default，只有 example_value
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

class XLAMLoader:
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
                    
                    # 判断是否可选：只看 type 字段中是否包含 'optional'
                    # 忠实于源数据，不因为有 default 就认为是 optional
                    is_optional = 'optional' in param_type.lower()
                    
                    param = Parameter(
                        name=param_name,
                        type=param_type.split(',')[0].strip(),  # 去掉 ", optional" 部分
                        description=param_info.get('description', ''),
                        default=param_info.get('default'),
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
    
    # 解析第 1 条，完整展示
    sample = loader.parse_record(loader.data[0], 0)
    if sample:
        print_sample(sample)
    
    print("\n" + "=" * 80)
    print("Testing xLAM Loader")
    print("=" * 80)
    
    xlam_path = '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/xlam_60k.jsonl'
    loader = XLAMLoader(xlam_path)
    loader.load()
    
    # 解析第 1 条，完整展示
    sample = loader.parse_record(loader.data[0], 0)
    if sample:
        print_sample(sample)
