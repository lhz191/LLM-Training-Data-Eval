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
